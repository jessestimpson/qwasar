defmodule Warden.Edit do
  @moduledoc """
  Line-anchored search and replace, matching `qw_edit_apply` exactly.

  PLAN.md 7.1: the `edit` tool's description is the C agent's, verbatim, and
  `Tests/golden.tsv` pins it into the system turn. The model is therefore told
  the C implementation's contract, so this has to *be* that contract — a guest
  that quietly accepted an ambiguous match, or matched a fragment of a line,
  would be doing something other than what the model was promised.

  The rules, and why each one is there:

    * **Whole lines only.** Anchoring to line boundaries is what stops a quoted
      fragment matching the middle of a longer line and producing a mangled
      result.
    * **Exactly once.** Not the first match — no match at all. A bare `}` is
      exactly the anchor that guessing gets wrong, and the model is told to
      quote more surrounding lines instead.
    * **A trailing newline on `old` is presentation, not content**, so quoting a
      block with or without one behaves the same.
    * **Deleting takes the line terminator with it**, or a deletion would leave
      a blank line where the text was.
    * **A file with no trailing newline neither gains nor loses one.**

  `test/warden_edit_test.exs` is the C suite's cases, ported one for one.
  """

  @type status :: :ok | :not_found | :ambiguous | :empty_old

  @doc """
  Replaces the unique run of whole lines matching `old` with `new`.

  Returns `{:ok, content, 1}`, or `{status, matches}` where `matches` is how
  many places matched — which is what the agent tells the model when the edit is
  refused.
  """
  @spec apply_edit(String.t(), String.t(), String.t()) ::
          {:ok, String.t(), 1} | {status(), non_neg_integer()}
  def apply_edit(content, old, new) do
    old = String.replace_trailing(old || "", "\n", "")

    if old == "" do
      {:empty_old, 0}
    else
      file_lines = split_lines(content)
      # The old text as strings, not spans: the file's lines stay as spans so
      # the splice can address the original bytes, but the comparison is
      # between text and text.
      old_lines = split_lines(old) |> Enum.map(&text(old, &1))
      match(content, file_lines, old_lines, new)
    end
  end

  defp match(content, file_lines, old_lines, new) do
    no = length(old_lines)
    nf = length(file_lines)

    found =
      if no > nf do
        []
      else
        # Stops at two: ambiguity is decided, and there is nothing further to
        # learn from a third.
        Enum.reduce_while(0..(nf - no), [], fn i, acc ->
          if Enum.slice(file_lines, i, no) |> Enum.map(&text(content, &1)) == old_lines do
            acc = [i | acc]
            if length(acc) > 1, do: {:halt, acc}, else: {:cont, acc}
          else
            {:cont, acc}
          end
        end)
      end

    case found do
      [] -> {:not_found, 0}
      [_, _ | _] -> {:ambiguous, 2}
      [at] -> {:ok, splice(content, file_lines, at, no, new), 1}
    end
  end

  defp splice(content, file_lines, at, no, new) do
    {a, _} = Enum.at(file_lines, at)
    {off, len} = Enum.at(file_lines, at + no - 1)
    b = off + len

    # An empty replacement takes the newline with it, so a deleted line leaves
    # nothing rather than a blank.
    b =
      if new == "" and b < byte_size(content) and binary_part(content, b, 1) == "\n",
        do: b + 1,
        else: b

    binary_part(content, 0, a) <> new <> binary_part(content, b, byte_size(content) - b)
  end

  defp text(content, {off, len}), do: binary_part(content, off, len)

  # Lines as {offset, length}, with the terminator excluded. A trailing newline
  # does not produce a final empty line, so "a\nb\n" and "a\nb" both split into
  # two -- which is what makes the whole-file case work either way.
  defp split_lines(s), do: split_lines(s, 0, 0, [])

  defp split_lines(s, i, start, acc) when i >= byte_size(s) do
    acc = if i > start, do: [{start, i - start} | acc], else: acc
    Enum.reverse(acc)
  end

  defp split_lines(s, i, start, acc) do
    case binary_part(s, i, 1) do
      "\n" -> split_lines(s, i + 1, i + 1, [{start, i - start} | acc])
      _ -> split_lines(s, i + 1, start, acc)
    end
  end

  @doc "What the model is told when an edit is refused."
  def status_text(:not_found), do: "the old text was not found in the file"
  def status_text(:ambiguous), do: "the old text matches more than one place"
  def status_text(:empty_old), do: "the old text is empty"
  def status_text(:ok), do: "ok"
end
