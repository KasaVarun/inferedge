#include <cstdlib>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <string>

TEST(OnnxRuntimeSmokeTest, ReportsThePinnedVersion) {
    ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0);
    const char* telemetry = std::getenv("ORT_DISABLE_TELEMETRY");
    ASSERT_NE(telemetry, nullptr);
    EXPECT_STREQ(telemetry, "1");

    try {
        {
            const Ort::Env runtime{ORT_LOGGING_LEVEL_WARNING, "inferedge"};
            EXPECT_EQ(Ort::GetVersionString(), "1.31.0");
        }
    } catch (const Ort::Exception& error) {
        FAIL() << error.what();
    }
}
