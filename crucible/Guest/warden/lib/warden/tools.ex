defmodule Warden.Tools do
  @moduledoc """
  The six file tools, running in the guest against `/work`.

  PLAN.md 12, M3. These replace the host-side stand-ins M1 shipped, and the
  point of the milestone is what they *cannot* do: `/work` is the guest's own
  copy of the project, so `write`, `edit` and `shell` change nothing the user
  can see. Useless and safe, which is the correct order — M5 adds the patch that
  crosses back, under approval.

  Two contracts are inherited rather than invented, and both are pinned:

    * The tool **descriptions** are the C agent's, verbatim, and
      `Tests/golden.tsv` fixes them into the system turn. The model is told the
      C implementation's contract, so these have to be that contract.
    * `edit`'s matching rules live in `Warden.Edit`, whose test suite is the C
      suite's cases ported one for one.

  Result sizes are capped hard, and the reason is measured rather than assumed:
  prefill runs at ~32 tok/s (PLAN.md 2.5), so a tool result costs roughly a
  second per 100 bytes before the model can react to it. Being told a file is
  49 KB and that grep exists is worth far more than receiving 49 KB.
  """

  @root "/work"
  @max_result_bytes 8 * 1024
  @max_grep_matches 60
  @max_list_entries 300
  @shell_timeout_ms 120_000

  @skip_dirs ~w(.git node_modules .build build _build deps DerivedData .venv __pycache__)

  @type reply :: {:ok, String.t()} | {:error, String.t(), String.t()}

  # ---- read ----------------------------------------------------------------

  @spec read(map()) :: reply
  def read(%{"path" => path}) do
    with {:ok, full} <- safe(path) do
      cond do
        File.dir?(full) ->
          {:error, "io", "#{path} is a directory; use list"}

        true ->
          case File.read(full) do
            {:ok, ""} -> {:ok, "[the file is empty]"}
            {:ok, data} -> {:ok, truncate_file(data, path)}
            {:error, reason} -> {:error, "io", "cannot read #{path}: #{:file.format_error(reason)}"}
          end
      end
    end
  end

  def read(_), do: {:error, "args", "read requires a path"}

  # ---- write ---------------------------------------------------------------

  @spec write(map()) :: reply
  def write(%{"path" => path, "content" => content}) do
    with {:ok, full} <- safe(path, must_exist: false) do
      File.mkdir_p!(Path.dirname(full))

      case File.write(full, content) do
        :ok ->
          lines = content |> String.split("\n") |> length()
          {:ok, "wrote #{byte_size(content)} bytes (#{lines} lines) to #{path}"}

        {:error, reason} ->
          {:error, "io", "cannot write #{path}: #{:file.format_error(reason)}"}
      end
    end
  end

  def write(_), do: {:error, "args", "write requires a path and content"}

  # ---- edit ----------------------------------------------------------------

  @spec edit(map()) :: reply
  def edit(%{"path" => path, "old" => old, "new" => new}) do
    with {:ok, full} <- safe(path),
         {:ok, content} <- read_file(full, path) do
      case Warden.Edit.apply_edit(content, old, new) do
        {:ok, updated, _} ->
          case File.write(full, updated) do
            :ok -> {:ok, "edited #{path}"}
            {:error, r} -> {:error, "io", "cannot write #{path}: #{:file.format_error(r)}"}
          end

        {:ambiguous, n} ->
          # The count is what the model needs: it is being told to quote more
          # surrounding lines, not that the edit was impossible.
          {:error, "ambiguous",
           "#{Warden.Edit.status_text(:ambiguous)} (#{n} or more). " <>
             "Quote enough surrounding lines to be unique."}

        {status, _} ->
          {:error, to_string(status), Warden.Edit.status_text(status)}
      end
    end
  end

  def edit(_), do: {:error, "args", "edit requires a path, old and new"}

  # ---- list ----------------------------------------------------------------

  @spec list(map()) :: reply
  def list(args) do
    path = Map.get(args, "path", ".")

    with {:ok, full} <- safe(path) do
      case File.ls(full) do
        {:ok, []} ->
          {:ok, "[the directory is empty]"}

        {:ok, entries} ->
          shown = entries |> Enum.sort() |> Enum.take(@max_list_entries)
          rest = length(entries) - length(shown)

          body =
            Enum.map_join(shown, "\n", fn e ->
              p = Path.join(full, e)
              if File.dir?(p), do: e <> "/", else: "#{e}  #{file_size(p)}"
            end)

          {:ok, cap(if rest > 0, do: body <> "\n[#{rest} more entries]", else: body)}

        {:error, reason} ->
          {:error, "io", "cannot list #{path}: #{:file.format_error(reason)}"}
      end
    end
  end

  # ---- grep ----------------------------------------------------------------

  @spec grep(map()) :: reply
  def grep(%{"pattern" => pattern} = args) do
    path = Map.get(args, "path", ".")

    with {:ok, full} <- safe(path) do
      # `grep -rEn`, matching the C agent and the description the model was
      # given: extended regular expressions, not ripgrep's dialect. -I skips
      # binaries, which is what keeps a checkout's object files out of the
      # context window.
      case System.cmd("grep", ["-rEnI", "--", pattern, full], stderr_to_stdout: true) do
        {out, status} when status in [0, 1] ->
          hits =
            out
            |> String.split("\n", trim: true)
            |> Enum.reject(&skipped?/1)
            |> Enum.map(&relative/1)

          shown = Enum.take(hits, @max_grep_matches)

          cond do
            shown == [] ->
              {:ok, "[no matches]"}

            length(hits) > @max_grep_matches ->
              {:ok, cap(Enum.join(shown, "\n") <> "\n[stopped at #{@max_grep_matches} matches; narrow the pattern]")}

            true ->
              {:ok, cap(Enum.join(shown, "\n"))}
          end

        {out, _} ->
          {:error, "grep", "not a valid regular expression: #{String.trim(out)}"}
      end
    end
  end

  def grep(_), do: {:error, "args", "grep requires a pattern"}

  # ---- shell ---------------------------------------------------------------

  @spec shell(map()) :: reply
  def shell(%{"command" => command}) do
    # No confirmation, deliberately. PLAN.md 7.1: inside the sandbox there is
    # nothing to protect -- the guest has no network device and cannot reach the
    # user's files. The confirmation moved to the boundary where it means
    # something, which is the patch in M5.
    task =
      Task.async(fn ->
        System.cmd("/bin/sh", ["-c", command],
          cd: @root,
          stderr_to_stdout: true,
          env: [{"PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}]
        )
      end)

    case Task.yield(task, @shell_timeout_ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, {out, 0}} ->
        {:ok, cap(if out == "", do: "[no output]", else: out)}

      {:ok, {out, code}} ->
        {:ok, cap("[exit #{code}]\n" <> out)}

      nil ->
        {:error, "timeout", "the command ran for #{div(@shell_timeout_ms, 1000)}s and was killed"}
    end
  rescue
    e -> {:error, "shell", Exception.message(e)}
  end

  def shell(_), do: {:error, "args", "shell requires a command"}

  # ---- confinement ---------------------------------------------------------

  # The guest-side half of a rule the host also enforces (PLAN.md 7.4 step 8).
  # Both sides check, because either one being wrong is a bug and neither is a
  # reason to trust the other.
  defp safe(path, opts \\ []) do
    full = Path.expand(path, @root)

    cond do
      not (full == @root or String.starts_with?(full, @root <> "/")) ->
        {:error, "path", "path escapes #{@root}: #{path}"}

      Keyword.get(opts, :must_exist, true) and not File.exists?(full) ->
        {:error, "path", "no such file or directory: #{path}"}

      true ->
        {:ok, full}
    end
  end

  defp read_file(full, path) do
    case File.read(full) do
      {:ok, data} -> {:ok, data}
      {:error, r} -> {:error, "io", "cannot read #{path}: #{:file.format_error(r)}"}
    end
  end

  defp file_size(p) do
    case File.stat(p) do
      {:ok, %{size: s}} -> s
      _ -> 0
    end
  end

  defp skipped?(line) do
    Enum.any?(@skip_dirs, &String.contains?(line, "/#{&1}/"))
  end

  defp relative("/work/" <> rest), do: rest
  defp relative(line), do: line

  # ---- ceilings ------------------------------------------------------------

  defp truncate_file(data, path) do
    if byte_size(data) <= @max_result_bytes do
      data
    else
      head = binary_part(data, 0, @max_result_bytes)
      # Cut on a line boundary: half a line of source is worse than none, and
      # the model quotes what it reads back into an edit.
      head =
        case :binary.matches(head, "\n") do
          [] -> head
          ms -> binary_part(head, 0, ms |> List.last() |> elem(0))
        end

      shown = head |> :binary.matches("\n") |> length()
      total = data |> :binary.matches("\n") |> length()

      head <>
        "\n\n[truncated: showed #{shown} of #{total} lines (#{byte_size(head)} of " <>
        "#{byte_size(data)} bytes) of #{path}. Reading the whole file would cost " <>
        "minutes of prompt processing. Use grep to find the lines you need.]"
    end
  end

  # The backstop: no tool can put more than one cap's worth of prefill in front
  # of the model, however it is called.
  defp cap(s) do
    if byte_size(s) <= @max_result_bytes do
      s
    else
      head = binary_part(s, 0, @max_result_bytes)
      head <> "\n[truncated at #{byte_size(head)} of #{byte_size(s)} bytes]"
    end
  end
end
