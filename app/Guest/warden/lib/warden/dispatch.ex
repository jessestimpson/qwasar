defmodule Warden.Dispatch do
  @moduledoc """
  What the host can ask for.

  Deliberately narrow at M2: enough to prove the channel end to end and nothing
  that writes. The six file tools move here in M3, and `define`/`invoke` in M4.

  Pure with respect to the bridge: it takes a decoded request and returns a
  reply, which keeps the correlation logic in one place and the contract
  testable without a socket.
  """

  @type reply :: {:ok, String.t()} | {:error, String.t(), String.t()}

  @spec run(map()) :: reply
  def run(%{"op" => "ping"}), do: {:ok, "pong"}

  def run(%{"op" => "info"}) do
    {:ok,
     Enum.join(
       [
         "otp=#{:erlang.system_info(:otp_release)}",
         "erts=#{:erlang.system_info(:version)}",
         "elixir=#{System.version()}",
         "kernel=#{cmd("uname", ["-r"])}",
         "runtimes=#{runtimes()}",
         "work_files=#{cmd("sh", ["-c", "find /work -type f 2>/dev/null | wc -l"])}",
         "schedulers=#{:erlang.system_info(:schedulers_online)}",
         "memory_mb=#{div(:erlang.memory(:total), 1_048_576)}"
       ],
       " "
     )}
  end

  # The six file tools (PLAN.md 12, M3). `op` is the tool's own name, so the
  # wire carries what the model asked for rather than a wrapper around it, and
  # confinement to /work lives in Warden.Tools where every one of them shares it.
  def run(%{"op" => "read", "args" => args}), do: Warden.Tools.read(args)
  def run(%{"op" => "write", "args" => args}), do: Warden.Tools.write(args)
  def run(%{"op" => "edit", "args" => args}), do: Warden.Tools.edit(args)
  def run(%{"op" => "list", "args" => args}), do: Warden.Tools.list(args)
  def run(%{"op" => "list"}), do: Warden.Tools.list(%{})
  def run(%{"op" => "grep", "args" => args}), do: Warden.Tools.grep(args)
  # `bash` is the name in the schema the model is given -- it is the C agent's,
  # verbatim, and the golden vectors pin it. `shell` is the same tool under the
  # name the rest of this codebase uses. Both, because a mismatch here means the
  # model calls a tool it was told exists and is told it does not.
  def run(%{"op" => "bash", "args" => args}), do: Warden.Tools.shell(args)
  def run(%{"op" => "shell", "args" => args}), do: Warden.Tools.shell(args)

  # Called with no arguments at all.
  #
  # Without these the op falls through to the catch-all and the model is told
  # "no such op: read" -- that the tool does not exist, when in fact it forgot a
  # parameter. Wrong diagnosis, and the model acts on the diagnosis: the run
  # that found this went looking for a JSON library because it had been told a
  # tool it had just been given did not exist.
  def run(%{"op" => "read"}), do: Warden.Tools.read(%{})
  def run(%{"op" => "write"}), do: Warden.Tools.write(%{})
  def run(%{"op" => "edit"}), do: Warden.Tools.edit(%{})
  def run(%{"op" => "grep"}), do: Warden.Tools.grep(%{})
  def run(%{"op" => op}) when op in ["bash", "shell"], do: Warden.Tools.shell(%{})

  # The agent's node (PLAN.md 7.3). `workspace` reports whether it is up;
  # `eval` runs Elixir on it, which is the seed of M4's `elixir` tool.
  def run(%{"op" => "workspace"}), do: {:ok, Warden.Workspace.status()}

  # `elixir` is the name in the schema; `eval` is the same thing under the name
  # this codebase used first. Both, for the same reason `bash` and `shell` are
  # both here: a name the model is given and cannot call is worse than no tool.
  def run(%{"op" => op, "args" => %{"code" => code}}) when op in ["elixir", "eval"] do
    Warden.Workspace.eval(code, 15_000)
  end

  def run(%{"op" => op}) when op in ["elixir", "eval"],
    do: {:error, "args", "elixir requires code"}

  # ---- self-modification (PLAN.md 7.2) ------------------------------------

  def run(%{"op" => "define", "args" => %{"source" => source} = args}) do
    Warden.Define.run(source, force: args["force"] in [true, "true", "1"])
  end

  def run(%{"op" => "define"}), do: {:error, "args", "define requires source"}

  def run(%{"op" => "skills"}) do
    case Warden.Registry.list() do
      [] ->
        {:ok, "[no skills defined yet — write one with define]"}

      entries ->
        {:ok,
         Enum.map_join(entries, "\n", fn e ->
           schema = if map_size(e.schema) == 0, do: "", else: " " <> inspect(e.schema)
           flag = if e.concurrent, do: " [concurrent]", else: ""
           "#{e.tool_name}  (#{e.module} v#{e.version})#{flag}#{schema}"
         end)}
    end
  end

  def run(%{"op" => "invoke", "args" => %{"name" => name} = args}) do
    case Warden.Registry.lookup(name) do
      nil ->
        {:error, "unknown_skill",
         "no skill named #{inspect(name)}. Call skills to see what is defined."}

      entry ->
        Warden.Workspace.invoke(entry.module, tool_args(args), 60_000)
    end
  end

  def run(%{"op" => "invoke"}), do: {:error, "args", "invoke requires a name"}

  # ---- the .git shadow (spec 7.4) ------------------------------------------
  #
  # Host-only, off the model's surface: arms a re-seed of the shadow for the
  # NEXT boot by dropping the seed stamp mount-work checks. A running guest
  # cannot see the real .git -- the bind mount is the point -- so refresh is
  # a reboot: the host parks the session after this, and the next boot
  # re-seeds while nothing but init is running.
  def run(%{"op" => "git_refresh"}) do
    case File.rm("/var/lib/crucible/git-seeded") do
      :ok -> {:ok, "armed: the shadow re-seeds on the next boot"}
      {:error, :enoent} -> {:ok, "no shadow was seeded; nothing to refresh"}
      {:error, r} -> {:error, "git_refresh", "could not drop the seed stamp: #{r}"}
    end
  end

  # Deliberately available: the only honest way to test that a workspace crash
  # is survivable is to cause one. It kills the agent's node, never warden's.
  def run(%{"op" => "workspace_kill"}) do
    case Warden.Workspace.eval("System.halt(1)", 2_000) do
      _ -> {:ok, "asked the workspace to halt"}
    end
  end

  # Handled by the worker, which defers it; it exists so a test can make a
  # request outlive its deadline. Listed here so an unknown-op reply never
  # claims it does not exist.
  def run(%{"op" => "sleep"}), do: {:ok, "slept"}

  def run(%{"op" => op}), do: {:error, "unknown_op", "no such op: #{op}"}

  def run(_), do: {:error, "malformed", "a request needs an op"}

  # ---- helpers -------------------------------------------------------------

  # `invoke` carries a tool's own arguments inside the envelope, and the model's
  # tool-call format is XML with no nested objects -- so what arrives is a JSON
  # *string*, not a map. Decoding it here is not a convenience: without it the
  # model's first invoke fails with "no function clause matching in
  # Tool.run/1", the argument having been wrapped as %{"input" => "{...}"},
  # which tells it nothing about what went wrong.
  #
  # A string that is not JSON is passed through under "input" rather than
  # rejected, because a single-argument tool taking a bare string is a
  # reasonable thing for the model to have written.
  defp tool_args(args) do
    case Map.get(args, "args", %{}) do
      m when is_map(m) ->
        m

      s when is_binary(s) ->
        try do
          case :json.decode(s) do
            m when is_map(m) -> m
            other -> %{"input" => other}
          end
        rescue
          _ -> %{"input" => s}
        end

      other ->
        %{"input" => other}
    end
  end

  # What the agent has to work with: the runtimes baked into the image,
  # reported by asking each one. There is no runtime installer in the guest --
  # a VM with no network device could never fetch one anyway, which is what
  # retired mise from this image (PLAN.md 6.2).
  #
  # Absolute paths on purpose: this runs under an init whose PATH is whatever
  # the kernel handed PID 1, and a PATH lookup that fails here would report
  # "no runtimes" for a guest that has them.
  defp runtimes do
    [
      {"erlang", "/usr/bin/erl",
       ["-noshell", "-eval", "io:put_chars(erlang:system_info(otp_release)), halt(0)."]},
      {"elixir", "/usr/bin/elixir", ["-e", "IO.write(System.version())"]},
      {"node", "/usr/bin/node", ["--version"]},
      {"python", "/usr/bin/python3", ["--version"]}
    ]
    |> Enum.filter(fn {_, path, _} -> File.exists?(path) end)
    |> Enum.map_join(",", fn {name, path, args} ->
      "#{name}@#{cmd(path, args) |> String.replace_prefix("Python ", "") |> String.replace_prefix("v", "")}"
    end)
    |> case do
      "" -> "none"
      s -> s
    end
  end

  defp cmd(exe, args) do
    case System.cmd(exe, args, stderr_to_stdout: true) do
      {out, 0} -> String.trim(out)
      {out, code} -> "(#{code}: #{String.trim(out)})"
    end
  rescue
    _ -> "(unavailable)"
  end
end
