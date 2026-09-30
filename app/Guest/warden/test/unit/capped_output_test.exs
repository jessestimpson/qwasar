defmodule Warden.CappedOutputTest do
  use ExUnit.Case, async: true

  test "keeps everything under the cap" do
    {out, 0} = System.cmd("printf", ["abc"], into: %Warden.CappedOutput{max: 10})
    assert Warden.CappedOutput.text(out) == "abc"
  end

  test "keeps the first max bytes and says how many it dropped" do
    {out, 0} = System.cmd("sh", ["-c", "yes | head -c 5000"], into: %Warden.CappedOutput{max: 100})
    text = Warden.CappedOutput.text(out)
    assert String.starts_with?(text, String.duplicate("y\n", 50))
    assert text =~ "[output stopped here; 4900 more bytes]"
  end
end
