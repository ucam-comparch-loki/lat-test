// Test the convolution network layer.

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <loki/alloc.h>
#include <loki/channel_map_table.h>
#include <nn/layers.h>

// Create an activation tensor. Assumes the default CPU memory group should be
// used. The user is responsible for `loki_free`ing the tensor.
// Dimension order is BCHW.
activation_config_t* init_activations(data_t* data, int batch_size,
                                      int channels, int height, int width) {
  activation_config_t* a = loki_malloc(sizeof(activation_config_t));
  assert(a != NULL);

  a->memoryConfigEncoded = get_channel_map(1);
  a->address = data;
  a->rowSkip = sizeof(data_t);
  a->columnSkip = width * a->rowSkip;
  a->channelSkip = height * a->columnSkip;
  a->batchSkip = channels * a->channelSkip;

  return a;
}

// Create a weight tensor. Assumes the default CPU memory group should be
// used. The user is responsible for `loki_free`ing the tensor.
// Dimension order is OIHW.
filter_config_t* init_weights(data_t* data, int in_channels, int out_channels,
                              int filter_height, int filter_width) {
  filter_config_t* f = loki_malloc(sizeof(filter_config_t));
  assert(f != NULL);

  f->memoryConfigEncoded = get_channel_map(1);
  f->address = data;
  f->rowSkip = sizeof(data_t);
  f->columnSkip = filter_width * f->rowSkip;
  f->inChannelSkip = filter_height * f->columnSkip;
  f->outChannelSkip = in_channels * f->inChannelSkip;

  // TODO: groupSkip. Not used for anything in this test.

  return f;
}


// No weights or activations. No compute should take place.
bool test_conv_empty() {
  // Don't expect any data to be used, but provide some known values so we can
  // detect if they are used.
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 0, 0, 0, 0);
  filter_config_t* weights = init_weights(dummy, 0, 0, 0, 0);
  conv_shape_t conv = {
    .batchSize = 0, .inChannels = 0, .outChannels = 0, .imageWidth = 0,
    .imageHeight = 0, .filterWidth = 0, .filterHeight = 0, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 2, 2, 2, 2);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No weights. No compute should take place.
bool test_conv_no_weights() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 2, 2, 2, 2);
  filter_config_t* weights = init_weights(dummy, 0, 0, 0, 0);
  conv_shape_t conv = {
    .batchSize = 2, .inChannels = 2, .outChannels = 0, .imageWidth = 2,
    .imageHeight = 2, .filterWidth = 0, .filterHeight = 0, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 2, 2, 2, 2);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No activations. No compute should take place.
bool test_conv_no_activations() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 0, 0, 0, 0);
  filter_config_t* weights = init_weights(dummy, 2, 2, 2, 2);
  conv_shape_t conv = {
    .batchSize = 0, .inChannels = 0, .outChannels = 2, .imageWidth = 0,
    .imageHeight = 0, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 2, 2, 2, 2);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No batch elements. No compute should take place.
bool test_conv_no_batch() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 0, 1, 4, 4);
  filter_config_t* weights = init_weights(dummy, 4, 1, 2, 2);
  conv_shape_t conv = {
    .batchSize = 0, .inChannels = 1, .outChannels = 4, .imageWidth = 4,
    .imageHeight = 4, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 0, 4, 3, 3);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No input channels. No compute should take place.
bool test_conv_no_in_channels() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 4, 0, 2, 2);
  filter_config_t* weights = init_weights(dummy, 2, 0, 2, 2);
  conv_shape_t conv = {
    .batchSize = 4, .inChannels = 0, .outChannels = 2, .imageWidth = 2,
    .imageHeight = 2, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 4, 2, 1, 1);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No output channels. No compute should take place.
bool test_conv_no_out_channels() {
  // Don't expect any data to be used, but provide some known values so we can
  // detect if they are used.
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 1, 4, 2, 2);
  filter_config_t* weights = init_weights(dummy, 0, 4, 2, 2);
  conv_shape_t conv = {
    .batchSize = 1, .inChannels = 4, .outChannels = 0, .imageWidth = 2,
    .imageHeight = 2, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 1, 0, 1, 1);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No input width. No compute should take place.
bool test_conv_no_width() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 2, 2, 4, 0);
  filter_config_t* weights = init_weights(dummy, 2, 2, 2, 2);
  conv_shape_t conv = {
    .batchSize = 2, .inChannels = 2, .outChannels = 2, .imageWidth = 0,
    .imageHeight = 4, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 2, 2, 3, 0);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}

// No input height. No compute should take place.
bool test_conv_no_height() {
  data_t dummy[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  activation_config_t* input = init_activations(dummy, 2, 2, 0, 4);
  filter_config_t* weights = init_weights(dummy, 2, 2, 2, 2);
  conv_shape_t conv = {
    .batchSize = 2, .inChannels = 2, .outChannels = 2, .imageWidth = 4,
    .imageHeight = 0, .filterWidth = 2, .filterHeight = 2, .groups = 1
  };

  activation_config_t* output = init_activations(dummy, 2, 2, 0, 3);

  lat_conv2d(input, weights, output, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<16; i++)
    if (output->address[i] != i)
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output);

  return pass;
}


bool test_conv_1x1_small() {
  // Add two channels together by setting all weights to 1.

  data_t input_data[2*2*2] = {
    1,0,
    1,0,

    0,0,
    2,2
  };

  data_t weight_data[1*1*2*1] = {
    1,1
  };

  data_t expected[2*2*1] = {
    1+0,   0+0,
    1+2,   0+2
  };

  activation_config_t* input = init_activations(input_data, 1, 2, 2, 2);
  filter_config_t* weights = init_weights(weight_data, 2, 1, 1, 1);
  conv_shape_t conv = {
    .batchSize = 1, .inChannels = 2, .outChannels = 1, .imageWidth = 2,
    .imageHeight = 2, .filterWidth = 1, .filterHeight = 1, .groups = 1
  };

  activation_config_t* output = lat_conv2d_alloc(input, weights, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<4; i++)
    if (output->address[i] != expected[i])
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output->address);
  loki_free(output);

  return pass;
}

bool test_conv_3x3_small() {
  data_t input_data[2*4*4] = {
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,

    -1, -1, -1, -1,
    -1, -1, -1, -1,
    -1, -1, -1, -1,
    -1, -1, -1, -1
  };

  data_t weight_data[2*2*3*3] = {
    // Two filters applied to two input channels, creating one output channel.
    0,1,0,
    0,1,0,
    0,1,0,

    0,2,0,
    0,2,0,
    0,2,0,

    // Two filters applied to two input channels, creating one output channel.
    0,0,0,
    0,0,0,
    0,0,0,

    0,0,0,
    2,2,2,
    0,0,0
  };

  data_t expected[2*2*2] = {
    2+6+10-2-2-2,   3+7+11-2-2-2,
    6+10+14-2-2-2,  7+11+15-2-2-2,

    0+0+0-2-2-2,    0+0+0-2-2-2,
    0+0+0-2-2-2,    0+0+0-2-2-2
  };

  activation_config_t* input = init_activations(input_data, 1, 2, 4, 4);
  filter_config_t* weights = init_weights(weight_data, 2, 2, 3, 3);
  conv_shape_t conv = {
    .batchSize = 1, .inChannels = 2, .outChannels = 2, .imageWidth = 4,
    .imageHeight = 4, .filterWidth = 3, .filterHeight = 3, .groups = 1
  };

  activation_config_t* output = lat_conv2d_alloc(input, weights, &conv, 1, 1);

  bool pass = true;

  for (int i=0; i<8; i++)
    if (output->address[i] != expected[i])
      pass = false;

  loki_free(input);
  loki_free(weights);
  loki_free(output->address);
  loki_free(output);

  return pass;
}

void test_conv_forward_stride() {
/*
  data_t bias_data[2] = {10, 20};

  data_t input_data[3*3*2] = {
    1,1,  1,2,  1,3,
    1,4,  1,5,  1,6,
    1,7,  1,8,  1,9
  };

  data_t weight_data[1*1*2*2] = {
    1,1,  1,-1
  };

  // Stride = 2, so just get the corner points.
  data_t expected[2*2*2] = {
    10+1+1,20+1-1,   10+1+3,20+1-3,
    10+1+7,20+1-7,   10+1+9,20+1-9
  };

  lokinn_tensor* biases = conv_bias_tensor_create(bias_data, 2);

  lokinn_tensor* weights = conv_weight_tensor_create(weight_data,
                                                     WEIGHT_ROWS,         1,
                                                     WEIGHT_COLUMNS,      1,
                                                     WEIGHT_IN_CHANNELS,  2,
                                                     WEIGHT_OUT_CHANNELS, 2);

  lokinn_tensor* input = activation_tensor_create(input_data,
                                                  ACT_ROWS,     3,
                                                  ACT_COLUMNS,  3,
                                                  ACT_CHANNELS, 2);

  for (int algorithm = 0; algorithm < CONV_NUM_ALGORITHMS; algorithm++) {
    for (int sparse_acts = 0; sparse_acts <= 1; sparse_acts++) {
      for (int sparse_weights = 0; sparse_weights <= 1; sparse_weights++) {
        if (DEBUG)
          print_configuration("FW stride", sparse_acts, sparse_weights, algorithm);

        lokinn_layer* layer =
            create_conv_layer_x("conv", NULL, weights, biases, 2,
                                sparse_acts, sparse_weights, algorithm);
        layer->input = input;

        layer_initialise(layer, NULL);
        assert(layer->output->dimensions == 3);
        assert(tensor_dimension_size(layer->output, ACT_ROWS) == 2);
        assert(tensor_dimension_size(layer->output, ACT_COLUMNS) == 2);
        assert(tensor_dimension_size(layer->output, ACT_CHANNELS) == 2);

        layer->output->data = loki_malloc(layer->output->size * sizeof(data_t));
        layer->output->memory_managed = true;

        layer_forward(layer);
        for (int i=0; i<layer->output->size; i++)
          assert(layer->output->data[i] == expected[i]);

        layer_delete(layer);
      }
    }
  }

  tensor_delete(input);
  tensor_delete(weights);
  tensor_delete(biases);
*/
}

// Tests take no arguments, and return a bool which is `true` if the test
// passed.
typedef bool test_fn(void);

#define NUM_TESTS 10
test_fn* tests[NUM_TESTS] = {
  test_conv_empty,
  test_conv_no_weights,
  test_conv_no_activations,
  test_conv_no_batch,
  test_conv_no_in_channels,
  test_conv_no_out_channels,
  test_conv_no_width,
  test_conv_no_height,
  test_conv_1x1_small,
  test_conv_3x3_small
};

void run_test(test_fn* test, int id) {
  bool passed = test();

  if (!passed)
    exit(id);
}

void run_all_tests() {
  // There is no test 0.
  for (int id=1; id <= NUM_TESTS; id++)
    run_test(tests[id-1], id);
}

int main(int argc, char** argv) {

  if (argc > 1) {
    if (!strncmp(argv[1], "--test=", 7)) {
      char* arg = argv[1] + 7;
      int id = atoi(arg);

      if (id <= NUM_TESTS) {
        // There is no test 0.
        run_test(tests[id-1], id);
      }
      else {
        printf("Unknown test ID: %d\n", id);
        exit(-1);
      }
    }
    else {
      printf("Unknown argument: %s\n", argv[1]);
      exit(-1);
    }
  }
  else {
    run_all_tests();
  }

  return 0;

}
