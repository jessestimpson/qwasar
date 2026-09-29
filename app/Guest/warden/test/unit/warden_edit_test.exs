defmodule Warden.EditTest do
  @moduledoc """
  The C suite's edit cases, ported one for one from `tests/test_toolcall.c`.

  The model is told the C implementation's contract (PLAN.md 7.1: the tool
  description is verbatim, and the golden vectors pin it into the system turn).
  So the guest's implementation has to agree with it, case for case, or the
  model has been promised something the sandbox does not do.
  """
  use ExUnit.Case, async: true

  @file_a """
  #include <stdio.h>

  int add(int a, int b) {
      return a + b;
  }

  int sub(int a, int b) {
      return a - b;
  }
  """

  describe "a unique match" do
    test "replaces a body" do
      assert {:ok, out, 1} = Warden.Edit.apply_edit(@file_a, "    return a + b;", "    return b + a;")

      assert out ==
               "#include <stdio.h>\n\nint add(int a, int b) {\n    return b + a;\n}\n" <>
                 "\nint sub(int a, int b) {\n    return a - b;\n}\n"
    end

    test "replaces a multi-line run" do
      assert {:ok, out, 1} =
               Warden.Edit.apply_edit(
                 @file_a,
                 "int add(int a, int b) {\n    return a + b;\n}",
                 "int add(int a, int b) { return a + b; }"
               )

      assert out ==
               "#include <stdio.h>\n\nint add(int a, int b) { return a + b; }\n" <>
                 "\nint sub(int a, int b) {\n    return a - b;\n}\n"
    end

    test "a trailing newline on old is presentation, not content" do
      assert {:ok, _, 1} = Warden.Edit.apply_edit(@file_a, "    return a + b;\n", "    return 0;")
    end
  end

  describe "ambiguity is refused, not guessed" do
    test "two identical lines" do
      assert {:ambiguous, 2} =
               Warden.Edit.apply_edit("x = 1;\ny = 2;\nx = 1;\n", "x = 1;", "x = 3;")
    end

    test "a bare closing brace" do
      assert {:ambiguous, 2} = Warden.Edit.apply_edit(@file_a, "}", "} /* end */")
    end
  end

  describe "no match" do
    test "absent text" do
      assert {:not_found, 0} = Warden.Edit.apply_edit(@file_a, "int mul(int a, int b) {", "x")
    end

    test "a fragment of a line does not match" do
      assert {:not_found, 0} = Warden.Edit.apply_edit(@file_a, "return a + b", "return b + a")
    end

    test "indentation is content" do
      assert {:not_found, 0} = Warden.Edit.apply_edit(@file_a, "return a + b;", "return b + a;")
    end
  end

  describe "deletion and edges" do
    test "deleting a line takes its newline with it" do
      assert {:ok, "a\nc\n", 1} = Warden.Edit.apply_edit("a\nb\nc\n", "b", "")
    end

    test "the first line" do
      assert {:ok, "A\nb\nc\n", 1} = Warden.Edit.apply_edit("a\nb\nc\n", "a", "A")
    end

    test "the last line" do
      assert {:ok, "a\nb\nC\n", 1} = Warden.Edit.apply_edit("a\nb\nc\n", "c", "C")
    end

    test "a file with no trailing newline neither gains nor loses one" do
      assert {:ok, "a\nB", 1} = Warden.Edit.apply_edit("a\nb", "b", "B")
    end

    test "the whole file" do
      assert {:ok, "x\n", 1} = Warden.Edit.apply_edit("a\nb\n", "a\nb", "x")
    end

    test "an empty old is refused" do
      assert {:empty_old, 0} = Warden.Edit.apply_edit(@file_a, "", "x")
    end

    test "inserting via a unique anchor" do
      assert {:ok, "a\nb\nc\n", 1} = Warden.Edit.apply_edit("a\nc\n", "a", "a\nb")
    end
  end
end
