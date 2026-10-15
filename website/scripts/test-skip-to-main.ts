import assert from "node:assert/strict";
import test from "node:test";
import { MAIN_CONTENT_ID, skipToMainContent } from "../src/skipToMain.ts";

test("skip link preserves route/hash and moves keyboard focus", () => {
  const location = {
    pathname: "/performance",
    hash: "#/performance",
    href: "http://127.0.0.1:4173/inferedge/#/performance",
  };
  let focused = false;
  let preventCount = 0;
  const main = {
    tabIndex: 0,
    focus() {
      focused = true;
    },
  };
  const historyLength = 3;
  const ok = skipToMainContent(
    {
      preventDefault() {
        preventCount += 1;
      },
    },
    (id: string) => {
      assert.equal(id, MAIN_CONTENT_ID);
      return main;
    },
  );
  assert.equal(ok, true);
  assert.equal(preventCount, 1);
  assert.equal(focused, true);
  assert.equal(main.tabIndex, -1);
  assert.equal(location.pathname, "/performance");
  assert.equal(location.hash, "#/performance");
  assert.equal(location.href, "http://127.0.0.1:4173/inferedge/#/performance");
  assert.equal(historyLength, 3);
});

test("skip link does not throw when main is missing", () => {
  let preventCount = 0;
  const ok = skipToMainContent(
    {
      preventDefault() {
        preventCount += 1;
      },
    },
    () => null,
  );
  assert.equal(ok, false);
  assert.equal(preventCount, 1);
});
