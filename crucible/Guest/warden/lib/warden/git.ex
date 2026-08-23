defmodule Warden.Git do
  @moduledoc """
  The git crossing, guest half (spec 7.4a): the sandbox boundary is a network
  boundary, and git is a distributed VCS, so work crosses as OBJECTS -- the
  host writes what it receives as loose objects plus one ref, and the user
  merges a real branch in their real repo.

  This module only reads the repository state and packages objects; the one
  thing it writes is the safety commit (`export/0` commits an uncommitted
  remainder so nothing the agent did is lost) and the session-start choice
  (`include_local/exclude_local`, spec 7.4a's dirty-tree question).

  The base -- the commit `/work` arrived with, recorded by init before any
  agent ran -- bounds every export: the host by construction has everything
  at or below it, so `rev-list base..HEAD` is exactly the new material.
  """

  @work "/work"
  @base_file "/run/crucible-base-head"

  # ---- state ----------------------------------------------------------------

  @doc "repo? · dirty? · head · base, for the host to route on."
  def info do
    if repo?() do
      dirty = match?({out, 0} when out != "", git(["status", "--porcelain"]))
      head = head_sha()
      {:ok, "repo=true dirty=#{dirty} head=#{head || "none"} base=#{base_sha() || "none"}"}
    else
      {:ok, "repo=false"}
    end
  end

  # ---- the session-start choice (spec 7.4a) ---------------------------------

  @doc "The user chose to work on their uncommitted state: commit it, visibly theirs."
  def include_local do
    ensure_repo(fn ->
      case git(["status", "--porcelain"]) do
        {"", 0} ->
          {:ok, "the tree was clean; nothing to include"}

        {_, 0} ->
          with {_, 0} <- git(["add", "-A"]),
               {_, 0} <-
                 git([
                   "-c", "user.name=#{local_author()}",
                   "-c", "user.email=local@crucible.invalid",
                   "commit", "-q", "-m", "local changes before this session"
                 ]) do
            {:ok, "local changes committed as their own commit"}
          else
            {out, _} -> {:error, "git", "could not commit local changes: #{out}"}
          end

        {out, _} ->
          {:error, "git", out}
      end
    end)
  end

  @doc "The user chose a clean base: the uncommitted state never enters the session."
  def exclude_local do
    ensure_repo(fn ->
      with {_, 0} <- git(["reset", "--hard", "-q", "HEAD"]),
           {_, 0} <- git(["clean", "-fdq"]) do
        {:ok, "reset to HEAD; the uncommitted changes are not in this sandbox"}
      else
        {out, _} -> {:error, "git", out}
      end
    end)
  end

  # ---- export ---------------------------------------------------------------

  @doc """
  Commits any uncommitted remainder, then lists everything the host lacks:
  `base tip sha1 sha2 ...` -- every object reachable from HEAD but not from
  the base. The host pulls contents in batches through `objects/1`.
  """
  def export do
    ensure_repo(fn ->
      case base_sha() do
        nil ->
          {:error, "git", "no base commit was recorded at boot; cannot bound the export"}

        base ->
          commit_remainder()

          with tip when tip != nil <- head_sha(),
               {out, 0} <- git(["rev-list", "--objects", "#{base}..#{tip}"]) do
            shas =
              out
              |> String.split("\n", trim: true)
              |> Enum.map(&(&1 |> String.split(" ") |> hd()))
              |> Enum.uniq()

            {:ok, Enum.join([base, tip | shas], " ")}
          else
            nil -> {:error, "git", "the repository has no HEAD"}
            {out, _} -> {:error, "git", out}
          end
      end
    end)
  end

  @doc """
  Raw store-form contents for a space-separated list of shas, as lines of
  `sha type base64(content)` -- store form (`git cat-file <type>`), not the
  pretty-printed one, because the host re-hashes exactly what git will read
  and refuses anything that does not match its name.
  """
  def objects(shas) when is_binary(shas) do
    ensure_repo(fn ->
      lines =
        shas
        |> String.split(" ", trim: true)
        |> Enum.map(fn sha ->
          with {type, 0} <- git(["cat-file", "-t", sha]),
               type = String.trim(type),
               {content, 0} <- git_raw(["cat-file", type, sha]) do
            "#{sha} #{type} #{Base.encode64(content)}"
          else
            _ -> "#{sha} missing"
          end
        end)

      {:ok, Enum.join(lines, "\n")}
    end)
  end

  # ---- plumbing -------------------------------------------------------------

  defp commit_remainder do
    case git(["status", "--porcelain"]) do
      {"", 0} ->
        :ok

      {_, 0} ->
        git(["add", "-A"])

        git([
          "-c", "user.name=Crucible Agent",
          "-c", "user.email=agent@crucible.invalid",
          "commit", "-q", "-m", "uncommitted work at proposal time"
        ])

        :ok

      _ ->
        :ok
    end
  end

  defp repo?, do: File.dir?(Path.join(@work, ".git"))

  defp ensure_repo(fun) do
    if repo?(), do: fun.(), else: {:error, "git", "#{@work} is not a git repository"}
  end

  defp head_sha do
    case git(["rev-parse", "HEAD"]) do
      {out, 0} -> String.trim(out)
      _ -> nil
    end
  end

  defp base_sha do
    case File.read(@base_file) do
      {:ok, s} ->
        t = String.trim(s)
        if t == "", do: nil, else: t

      _ ->
        nil
    end
  end

  defp local_author do
    "Local changes"
  end

  defp git(args) do
    {out, code} = System.cmd("git", args, cd: @work, stderr_to_stdout: true)
    {out, code}
  end

  # Binary-safe: tree objects are raw bytes, and `stderr_to_stdout` plus
  # to_string would mangle them.
  defp git_raw(args) do
    {out, code} = System.cmd("git", args, cd: @work)
    {out, code}
  end
end
