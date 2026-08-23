defmodule Warden.Materialise do
  @moduledoc """
  What changed in `/work`, in a form the host can review and apply.

  PLAN.md 7.4. This is the only thing that crosses back out of the sandbox, and
  it crosses as a *proposal*: the host validates every path, checks the user's
  tree has not moved underneath, shows the change, and writes nothing without
  approval.

  ## Why the diff and the content are both sent

  The obvious design is to send a patch and have the host run `git apply`. That
  cannot work here. The host is sandboxed and reaches the user's folder through
  a security-scoped bookmark, and **that access does not survive into a child
  process** -- a spawned `git` would find the directory unreadable. So the host
  applies in-process, which means it needs the resulting bytes.

  The diff is therefore for the human and the content is for the machine. That
  also removes a whole class of risk: no patch headers to escape through, no
  `-p` level to get wrong, no three-way merge to reason about. The host writes
  the bytes it was given, to a path it has already validated, or it writes
  nothing.
  """

  @git_dir "/work/.crucible-git"
  @root "/work"
  @max_file_bytes 2 * 1024 * 1024
  @max_total_bytes 6 * 1024 * 1024

  @doc """
  Every change since the baseline, as JSON the host decodes.

  Reports its own limits rather than silently truncating: a proposal that
  quietly omitted a file would be the worst possible failure here.
  """
  def propose do
    with :ok <- have_baseline() do
      case status() do
        {:ok, []} ->
          {:ok, encode(%{changes: [], skipped: [], summary: "no changes"})}

        {:ok, entries} ->
          {changes, skipped, _} =
            Enum.reduce(entries, {[], [], 0}, fn {code, path}, {acc, skip, total} ->
              case change(code, path, total) do
                {:ok, c, n} -> {[c | acc], skip, total + n}
                {:skip, why} -> {acc, [%{path: path, reason: why} | skip], total}
              end
            end)

          changes = Enum.reverse(changes)

          {:ok,
           encode(%{
             changes: changes,
             skipped: Enum.reverse(skipped),
             summary: summarise(changes, skipped)
           })}

        {:error, msg} ->
          {:error, "git", msg}
      end
    end
  end

  @doc """
  Marks the current state as the new baseline.

  Called by the host after a proposal is applied, so the next one is a diff
  against what the user actually accepted rather than against the original.
  """
  def accept do
    with :ok <- have_baseline(),
         {_, 0} <- git(["add", "-A"]),
         {_, 0} <- git(["commit", "-q", "-m", "accepted", "--allow-empty"]) do
      {:ok, "baseline moved to the accepted state"}
    else
      {out, code} -> {:error, "git", "re-baselining failed (#{code}): #{String.trim(out)}"}
      other -> other
    end
  end

  # ---- one change ----------------------------------------------------------

  defp change(code, path, total) do
    full = Path.join(@root, path)

    cond do
      code == "D" ->
        {:ok,
         compact(%{
           path: path,
           status: "deleted",
           diff: diff_for(path),
           content: nil,
           sha_before: sha_of_baseline(path),
           sha_after: nil
         }), 0}

      not File.regular?(full) ->
        {:skip, "not a regular file"}

      true ->
        case File.read(full) do
          {:error, reason} ->
            {:skip, "cannot read: #{reason}"}

          {:ok, data} when byte_size(data) > @max_file_bytes ->
            {:skip, "#{byte_size(data)} bytes exceeds the #{@max_file_bytes}-byte limit"}

          {:ok, data} ->
            if total + byte_size(data) > @max_total_bytes do
              {:skip, "the proposal would exceed #{@max_total_bytes} bytes in total"}
            else
              {:ok,
               compact(%{
                 path: path,
                 status: if(code == "A" or code == "??", do: "added", else: "modified"),
                 diff: diff_for(path),
                 # base64 so a binary file survives JSON intact.
                 content: Base.encode64(data),
                 sha_before: if(code in ["A", "??"], do: nil, else: sha_of_baseline(path)),
                 sha_after: sha256(data)
               }), byte_size(data)}
            end
        end
    end
  end

  # ---- git -----------------------------------------------------------------

  defp status do
    case git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]) do
      {out, 0} ->
        entries =
          out
          |> String.split(<<0>>, trim: true)
          |> Enum.flat_map(fn entry ->
            case String.split_at(entry, 3) do
              {code, path} when path != "" -> [{String.trim(code), path}]
              _ -> []
            end
          end)
          # The baseline's own git dir is not part of the project.
          |> Enum.reject(fn {_, p} -> String.starts_with?(p, ".crucible-git") end)

        {:ok, entries}

      {out, code} ->
        {:error, "git status failed (#{code}): #{String.trim(out)}"}
    end
  end

  defp diff_for(path) do
    case git(["diff", "HEAD", "--", path]) do
      {"", 0} ->
        # An untracked file has no diff against HEAD; show it as an addition.
        case git(["diff", "--no-index", "--", "/dev/null", Path.join(@root, path)]) do
          {out, _} -> out
        end

      {out, 0} ->
        out

      {out, _} ->
        out
    end
  end

  defp sha_of_baseline(path) do
    case git(["show", "HEAD:" <> path]) do
      {out, 0} -> sha256(out)
      _ -> nil
    end
  end

  defp have_baseline do
    if File.dir?(@git_dir) do
      :ok
    else
      {:error, "no_baseline",
       "this sandbox has no baseline, so changes cannot be proposed. " <>
         "The guest records one at boot; see the console log."}
    end
  end

  defp git(args) do
    System.cmd("git", args,
      cd: @root,
      stderr_to_stdout: true,
      env: [{"GIT_DIR", @git_dir}, {"GIT_WORK_TREE", @root}]
    )
  rescue
    e -> {Exception.message(e), 1}
  end

  # ---- reporting -----------------------------------------------------------

  defp summarise(changes, skipped) do
    by = Enum.frequencies_by(changes, & &1.status)

    parts =
      for s <- ["added", "modified", "deleted"], n = Map.get(by, s, 0), n > 0 do
        "#{n} #{s}"
      end

    base = if parts == [], do: "no changes", else: Enum.join(parts, ", ")
    if skipped == [], do: base, else: base <> ", #{length(skipped)} skipped"
  end

  defp sha256(data), do: :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

  # Drops keys whose value is absent, rather than sending them as null.
  #
  # OTP's :json encodes Elixir's `nil` -- which is the ATOM nil, not `null` --
  # as the string "nil". A deletion would therefore arrive at the host with
  # content: "nil", the host would fail to base64-decode it, and would refuse
  # the change as "the guest sent no content for it". An absent key decodes to
  # a Swift `nil` correctly, so absence is what is sent.
  defp compact(map), do: :maps.filter(fn _k, v -> v != nil end, map)

  defp encode(map), do: :erlang.iolist_to_binary(:json.encode(map))
end
