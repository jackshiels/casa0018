#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TfLite pointers
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Memory area for the tensor arrays and other numerical vals
const int kTensorArenaSize = (2 * 1024);
uint8_t tensor_arena[kTensorArenaSize];
const float kXrange = 2.f * 3.14159265359f;
const int kInferencesPerCycle = 48000;

// Track current inferences
int inference_count = 0;

// LED address
int led = LED_BUILTIN;

void setup() {
  // Set up the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  model = tflite::GetModel(models_model_tflite);

  // Check for validity
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
    "Model is incorrect schema.");
    return;
  }

  // Imports operations like ReLU, SoftMax...
  static tflite::AllOpsResolver ops_resolver;

  // Runs the model with an interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, ops_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk){
    TF_LITE_REPORT_ERROR(error_reporter, "Allocation failed.");
  }

  // Obtain I/O pointers
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Misc
  inference_count = 0;
  pinMode(led, OUTPUT);
}

void loop() {
  // Generate the x position
  float position = static_cast<float>(inference_count) / 
                    static_cast<float>(kInferencesPerCycle);

  // Multiply by the range (2pi)
  float x_val = position * kXrange;

  // Quantize the x_val
  int8_t x_quantized = x_val / 
                        input->params.scale + input->params.zero_point;
  input->data.int8[0] = x_quantized;

  // Run the model
  TfLiteStatus invoke_status = interpreter->Invoke();
  // Verify the status of the invocation
  if (invoke_status != kTfLiteOk){
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }

  // Get the value back out
  int8_t y_quantized = output->data.int8[0];
  float y_val = (y_quantized - output->params.zero_point) * output->params.scale;

  // Convert to brightness
  int brightness = (int)(127.5f * (y_val + 1));
  brightness = constrain(brightness, 0, 255);

  // Write to output
  analogWrite(led, brightness);

  // continue the cycle
  inference_count += 1;
  if (inference_count > kInferencesPerCycle) inference_count = 0;
}
