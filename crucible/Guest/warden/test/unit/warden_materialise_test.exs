defmodule Warden.MaterialiseTest do
  @moduledoc """
  The proposal, from the guest's side.

  PLAN.md 7.4. The host's half is tested adversarially in
  `Tests/MaterialiseSuite.swift`; this is the half that decides what the host is
  told about in the first place. A change the guest fails to report is a change
  the user never gets the chance to approve, which is the quietest way this
  could go wrong.

  Runs against the real `/work` with a real git baseline, because the baseline
  is the thing under test.
  """
  use ExUnit.Case, async: false
  # Writes under the real /work, which exists only in the guest. Excluded by
  # the image build (mkimage.sh runs on macOS); `make sandbox` exercises the
  # real tools in the real guest instead.
  @moduletag :work_fs

  @root "/work"
  @git "/work/.crucible-git"

  setup do
    File.rm_rf(@git)
    File.rm_rf(Path.join(@root, "mat"))
    File.mkdir_p!(Path.join(@root, "mat"))
    File.write!(Path.join(@root, "mat/kept.txt"), "unchanged\n")
    File.write!(Path.join(@root, "mat/edited.txt"), "before\n")
    File.write!(Path.join(@root, "mat/removed.txt"), "doomed\n")
    baseline()
    on_exit(fn ->
      File.rm_rf(@git)
      File.rm_rf(Path.join(@root, "mat"))
    end)
    :ok
  end

  defp baseline do
    env = [{"GIT_DIR", @git}, {"GIT_WORK_TREE", @root}]
    System.cmd("git", ["init", "-q"], cd: @root, env: env, stderr_to_stdout: true)
    System.cmd("git", ["config", "user.email", "t@localhost"], cd: @root, env: env)
    System.cmd("git", ["config", "user.name", "t"], cd: @root, env: env)
    System.cmd("git", ["add", "-A"], cd: @root, env: env, stderr_to_stdout: true)
    System.cmd("git", ["commit", "-q", "-m", "baseline", "--allow-empty"],
      cd: @root, env: env, stderr_to_stdout: true)
  end

  defp propose! do
    {:ok, json} = Warden.Materialise.propose()
    :json.decode(json)
  end

  defp change(%{"changes" => cs}, path), do: Enum.find(cs, &(&1["path"] == path))

  test "an untouched tree proposes nothing" do
    p = propose!()
    assert p["changes"] == []
    assert p["summary"] == "no changes"
  end

  test "a modification carries its content, its diff, and the baseline hash" do
    File.write!(Path.join(@root, "mat/edited.txt"), "after\n")
    p = propose!()

    c = change(p, "mat/edited.txt")
    assert c["status"] == "modified"
    assert Base.decode64!(c["content"]) == "after\n"
    # The hash is of the file as the SANDBOX first saw it, which is what lets
    # the host notice the user editing the same file meanwhile.
    assert c["sha_before"] == hash("before\n")
    assert c["sha_after"] == hash("after\n")
    assert c["diff"] =~ "-before"
    assert c["diff"] =~ "+after"
  end

  test "a new file is proposed as an addition with no baseline hash" do
    File.write!(Path.join(@root, "mat/fresh.txt"), "new\n")
    c = change(propose!(), "mat/fresh.txt")
    assert c["status"] == "added"
    assert c["sha_before"] == nil
    assert Base.decode64!(c["content"]) == "new\n"
  end

  test "a deletion is proposed with no content" do
    File.rm!(Path.join(@root, "mat/removed.txt"))
    c = change(propose!(), "mat/removed.txt")
    assert c["status"] == "deleted"
    assert c["content"] == nil
    assert c["sha_before"] == hash("doomed\n")
  end

  test "an untouched file is not proposed" do
    File.write!(Path.join(@root, "mat/edited.txt"), "after\n")
    assert change(propose!(), "mat/kept.txt") == nil
  end

  test "the baseline's own git directory is never proposed" do
    File.write!(Path.join(@root, "mat/edited.txt"), "after\n")
    p = propose!()
    refute Enum.any?(p["changes"], &String.starts_with?(&1["path"], ".crucible-git"))
  end

  test "accepting moves the baseline, so the next proposal is empty" do
    File.write!(Path.join(@root, "mat/edited.txt"), "after\n")
    assert length(propose!()["changes"]) == 1

    assert {:ok, _} = Warden.Materialise.accept()
    assert propose!()["changes"] == []

    # And a change after that is measured against the accepted state.
    File.write!(Path.join(@root, "mat/edited.txt"), "after again\n")
    c = change(propose!(), "mat/edited.txt")
    assert c["sha_before"] == hash("after\n")
  end

  test "a file too large to carry is reported as skipped, never omitted" do
    File.write!(Path.join(@root, "mat/huge.bin"), String.duplicate("x", 3 * 1024 * 1024))
    p = propose!()
    assert change(p, "mat/huge.bin") == nil
    skipped = Enum.find(p["skipped"], &(&1["path"] == "mat/huge.bin"))
    assert skipped, "a file that cannot be carried must be reported, not dropped"
    assert skipped["reason"] =~ "limit"
    assert p["summary"] =~ "skipped"
  end

  defp hash(s), do: :crypto.hash(:sha256, s) |> Base.encode16(case: :lower)
end
