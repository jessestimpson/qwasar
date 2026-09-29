defmodule Warden.ToolsTest do
  @moduledoc """
  Confinement and ceilings, from the guest's side.

  PLAN.md 7.4 step 8 makes path containment the security boundary and says both
  sides check it, because either one being wrong is a bug and neither is a
  reason to trust the other. The host's half has `Tests/PathGuardSuite.swift`;
  this is the guest's, and like that one it asserts on the *reason* a path was
  refused rather than merely that something failed.

  These run against the real `/work`, which exists in the guest and is created
  here when it does not — the tools have `/work` compiled in, and a test that
  pointed them somewhere else would be testing something else.
  """
  use ExUnit.Case, async: false
  # Writes under the real /work, which exists only in the guest. Excluded by
  # the image build (mkimage.sh runs on macOS); `make sandbox` exercises the
  # real tools in the real guest instead.
  @moduletag :work_fs

  @root "/work"

  setup do
    File.mkdir_p!(Path.join(@root, "sub"))
    File.write!(Path.join(@root, "hello.txt"), "one\ntwo\nthree\n")
    File.mkdir_p!("/tmp/outside")
    File.write!("/tmp/outside/secret.txt", "secret")
    on_exit(fn ->
      File.rm_rf(Path.join(@root, "sub"))
      File.rm(Path.join(@root, "hello.txt"))
    end)
    :ok
  end

  describe "paths are confined to /work" do
    test "an absolute path outside is refused" do
      assert {:error, "path", msg} = Warden.Tools.read(%{"path" => "/etc/hostname"})
      assert msg =~ "escapes /work"
    end

    test "a parent traversal is refused" do
      assert {:error, "path", msg} = Warden.Tools.read(%{"path" => "../tmp/outside/secret.txt"})
      assert msg =~ "escapes /work"
    end

    test "a traversal that first goes down is refused" do
      assert {:error, "path", msg} =
               Warden.Tools.read(%{"path" => "sub/../../tmp/outside/secret.txt"})

      assert msg =~ "escapes /work"
    end

    test "writing outside is refused too" do
      assert {:error, "path", _} =
               Warden.Tools.write(%{"path" => "../tmp/pwned.txt", "content" => "x"})

      refute File.exists?("/tmp/pwned.txt")
    end

    test "a sibling sharing the root's name prefix is refused" do
      File.mkdir_p!("/work-evil")
      assert {:error, "path", msg} = Warden.Tools.read(%{"path" => "../work-evil/x"})
      assert msg =~ "escapes /work"
    end

    test "a path inside is allowed" do
      assert {:ok, "one\ntwo\nthree\n"} = Warden.Tools.read(%{"path" => "hello.txt"})
    end
  end

  describe "the tools do what they say" do
    test "write then read round-trips" do
      assert {:ok, msg} = Warden.Tools.write(%{"path" => "sub/new.txt", "content" => "abc\n"})
      assert msg =~ "wrote 4 bytes"
      assert {:ok, "abc\n"} = Warden.Tools.read(%{"path" => "sub/new.txt"})
    end

    test "edit refuses an ambiguous match and says how many" do
      File.write!(Path.join(@root, "dup.txt"), "x = 1;\ny = 2;\nx = 1;\n")
      assert {:error, "ambiguous", msg} =
               Warden.Tools.edit(%{"path" => "dup.txt", "old" => "x = 1;", "new" => "x = 3;"})

      assert msg =~ "more than one place"
      # And the file is untouched.
      assert File.read!(Path.join(@root, "dup.txt")) == "x = 1;\ny = 2;\nx = 1;\n"
    end

    test "edit applies a unique match" do
      assert {:ok, _} =
               Warden.Tools.edit(%{"path" => "hello.txt", "old" => "two", "new" => "TWO"})

      assert File.read!(Path.join(@root, "hello.txt")) == "one\nTWO\nthree\n"
    end

    test "list shows entries" do
      assert {:ok, out} = Warden.Tools.list(%{"path" => "."})
      assert out =~ "hello.txt"
      assert out =~ "sub/"
    end

    test "grep finds a line and reports it relative to /work" do
      assert {:ok, out} = Warden.Tools.grep(%{"pattern" => "tw[o]", "path" => "."})
      assert out =~ "hello.txt:2:two"
      refute out =~ "/work/"
    end

    test "grep reports no matches rather than failing" do
      assert {:ok, "[no matches]"} =
               Warden.Tools.grep(%{"pattern" => "zzzznotpresent", "path" => "."})
    end

    test "shell runs in /work and reports a non-zero exit" do
      assert {:ok, out} = Warden.Tools.shell(%{"command" => "pwd && ls hello.txt"})
      assert out =~ "/work"
      assert {:ok, bad} = Warden.Tools.shell(%{"command" => "exit 3"})
      assert bad =~ "[exit 3]"
    end
  end

  describe "results are capped" do
    test "a large file is truncated on a line boundary, with the real size stated" do
      big = String.duplicate("a line of text that is quite long indeed\n", 500)
      File.write!(Path.join(@root, "big.txt"), big)
      assert {:ok, out} = Warden.Tools.read(%{"path" => "big.txt"})
      assert byte_size(out) < byte_size(big)
      assert out =~ "truncated"
      assert out =~ "#{byte_size(big)} bytes"
      # The cut is on a line boundary, so the last shown line is whole.
      assert out |> String.split("\n") |> Enum.at(0) == "a line of text that is quite long indeed"
    end
  end
end
