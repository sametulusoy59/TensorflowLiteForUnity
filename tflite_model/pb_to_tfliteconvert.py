import tensorflow as tf

# # Convert the model.
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    # graph_def_file='tflite_graph.pb',
                    # # both `.pb` and `.pbtxt` files are accepted.
    # input_arrays=['input'],
    # input_shapes={'input' : [1, 224, 224,3]},
    # output_arrays=['MobilenetV1/Predictions/Softmax']
# )
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
  # f.write(tflite_model)
  
  
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph("tflite_graph.pb",input_shapes = {'normalized_input_image_tensor':[1,300,300,3]},
    input_arrays = ['normalized_input_image_tensor'],output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',
    'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'])

converter.allow_custom_ops=True

# Convert the model to quantized TFLite model.
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


# Write a model using the following line
open("raspimodel.tflite", "wb").write(tflite_model)