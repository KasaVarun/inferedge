import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  ADDITIONAL_MODELS,
  CPP_CI,
  CPP_CI_RUN_URL,
  GEMM_MODEL,
  GEMM_ONNX_SHA256,
  GEMM_STUDY,
  RESNET18_MODEL,
  RESNET18_ONNX_SHA256,
  WEBSITE_CI,
  WEBSITE_CI_RUN_URL,
} from "../src/catalog/inferedgeEvidence.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "../..");
const cmake = readFileSync(join(root, "CMakeLists.txt"), "utf8");
const resnetHeader = readFileSync(join(root, "include/inferedge/resnet18_fixtures.hpp"), "utf8");

test("featured ONNX hashes match CMake and the ResNet header", () => {
  assert.match(cmake, new RegExp(GEMM_ONNX_SHA256));
  assert.match(cmake, new RegExp(RESNET18_ONNX_SHA256));
  assert.match(resnetHeader, new RegExp(RESNET18_ONNX_SHA256));
  assert.equal(GEMM_MODEL.onnxSha256, GEMM_ONNX_SHA256);
  assert.equal(RESNET18_MODEL.onnxSha256, RESNET18_ONNX_SHA256);
});

test("featured tensor contracts match the committed identities", () => {
  assert.equal(GEMM_MODEL.id, "gemm_f32");
  assert.equal(GEMM_MODEL.inputName, "features");
  assert.equal(GEMM_MODEL.inputShape, "[N, 512]");
  assert.equal(GEMM_MODEL.outputName, "product");
  assert.equal(GEMM_MODEL.outputShape, "[N, 512]");
  assert.equal(RESNET18_MODEL.id, "resnet18_f32");
  assert.equal(RESNET18_MODEL.inputName, "images");
  assert.equal(RESNET18_MODEL.inputShape, "[N, 3, 224, 224]");
  assert.equal(RESNET18_MODEL.outputName, "logits");
  assert.equal(RESNET18_MODEL.outputShape, "[N, 1000]");
  assert.equal(RESNET18_MODEL.fixtureIdMin, 1000);
  assert.equal(RESNET18_MODEL.fixtureIdMax, 1007);
  assert.equal(RESNET18_MODEL.atol, "1e-5");
  assert.equal(RESNET18_MODEL.rtol, "1e-5");
  assert.match(resnetHeader, /kResNet18FixtureIdMin = 1000/);
  assert.match(resnetHeader, /kResNet18FixtureIdMax = 1007/);
  assert.match(resnetHeader, /kResNet18AbsTol = 1.0e-5F/);
});

test("additional catalog models are CMake-pinned and have no latency extract", () => {
  for (const model of ADDITIONAL_MODELS) {
    assert.match(cmake, new RegExp(`${model.id}\\.onnx`));
    assert.equal(model.publishedLatency, false);
  }
});

test("published GEMM methodology constants stay aligned with the extract", () => {
  assert.equal(GEMM_STUDY.contrastId, "gemm_f32_G2_vs_G4_sparse_rps100");
  assert.equal(GEMM_STUDY.offeredRps, 100);
  assert.equal(GEMM_STUDY.durationMs, 5000);
  assert.equal(GEMM_STUDY.workers, 1);
  assert.equal(GEMM_STUDY.uncountedPilots, 2);
  assert.equal(GEMM_STUDY.countedBlocks, 20);
  assert.equal(GEMM_STUDY.replicates, 10000);
  assert.equal(GEMM_STUDY.seed, 20261016);
  assert.equal(GEMM_STUDY.coarseLagMisses, 3);
});

test("CI citations are specific successful runs", () => {
  assert.equal(CPP_CI.url, CPP_CI_RUN_URL);
  assert.equal(WEBSITE_CI.url, WEBSITE_CI_RUN_URL);
  assert.equal(CPP_CI.ctest, "800/800");
  assert.equal(CPP_CI.benchmarkPython, 537);
  assert.equal(CPP_CI.dynamicOnnx, 30);
  assert.equal(WEBSITE_CI.frontendTests, 52);
  assert.match(CPP_CI.url, /38840295859/);
  assert.match(WEBSITE_CI.url, /38840349770/);
});
