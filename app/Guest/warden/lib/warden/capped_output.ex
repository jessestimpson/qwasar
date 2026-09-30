defmodule Warden.CappedOutput do
  @moduledoc """
  A `Collectable` for `System.cmd/3` that keeps the first `max` bytes of a
  command's output and counts the rest, so `exec` holds a bounded amount
  however much a command prints.
  """

  defstruct max: 4_194_304, kept: [], size: 0, dropped: 0

  @doc "What was kept, with a note of what was not."
  def text(%__MODULE__{kept: kept, dropped: 0}), do: IO.iodata_to_binary(Enum.reverse(kept))

  def text(%__MODULE__{kept: kept, dropped: d}),
    do: IO.iodata_to_binary(Enum.reverse(kept)) <> "\n[output stopped here; #{d} more bytes]"

  defimpl Collectable do
    def into(acc) do
      fun = fn
        %{size: s, max: m} = a, {:cont, chunk} when s >= m ->
          %{a | dropped: a.dropped + byte_size(chunk)}

        %{size: s, max: m} = a, {:cont, chunk} ->
          room = m - s

          if byte_size(chunk) <= room do
            %{a | kept: [chunk | a.kept], size: s + byte_size(chunk)}
          else
            %{a | kept: [binary_part(chunk, 0, room) | a.kept], size: m,
                  dropped: a.dropped + byte_size(chunk) - room}
          end

        a, :done ->
          a

        _, :halt ->
          :ok
      end

      {acc, fun}
    end
  end
end
