# -*- coding: utf-8 -*-
"""Seb EfficientNet.ipynb


#Imports and Environment
"""

import os
if not os.path.exists('tpu'):
  !git clone https://github.com/tensorflow/tpu/

import sys
sys.path.append('/content/tpu/models/official/efficientnet')
sys.path.append('/content/tpu/models/common')
sys.path.append('/content/tpu/tools/datasets')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import json
import logging
from io import BytesIO


import numpy as np
import pandas as pd
import tensorflow as tf
print("Running on Tensorflow v{}".format(tf.__version__))
from tensorflow.python.tools import inspect_checkpoint as ic
import tensorflow.compat.v2 as tf2  # used for summaries only.

import main as efficientnet_main
import imagenet_input
import efficientnet_builder
import utils
import functools
from functools import partial
import xml.etree.ElementTree as ET


from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import google.auth

!pip install gcsfs

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

"""Set Logging Level"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  
logging.getLogger('tensorflow').setLevel(logging.INFO)
logging.disable(logging.NOTSET)
tf.get_logger().propagate = False

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
#logging.getLogger('tensorflow').setLevel(logging.ERROR)
#logging.disable(logging.WARNING)
#tf.get_logger().propagate = False

"""Connect to TPU"""

# Assert that notebook is connected to a TPU runtime
assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime'
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
print('TPU address is', TPU_ADDRESS)

# Connect with TPU with Google Credentials
from google.colab import auth
auth.authenticate_user()
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  print(session.list_devices())

  # Upload credentials to TPU.
  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
  # Now credentials are set for all future sessions on this TPU.

"""# Training Fully Connected Layer

###Parameters
"""

model_name = "efficientnet-b0" #@param ["efficientnet-b0", "efficientnet-b5"] {type:"string"}
data_set = "dataset_tf_10000-100" #@param ["dataset_tfrecords", "dataset_tf_10000-100", "cifar100", "dataset-tf-balanced"] {type:"string"}
run_name = "FT-14-01-20-4"

model_params = {
    "efficientnet-b0":{"iterations_per_loop":1024, "train_batch_size":1024},
    "efficientnet-b5":{"iterations_per_loop":32, "train_batch_size":128}
                }


data_set_params = {
    "dataset_tfrecords":{"num_train_images":33021, "num_eval_images":8257, "num_label_classes":96,"data_dir":"gs://ise-bucket/efficientnet/dataset_tf_records"},
    "dataset_tf_10000-100":{"num_train_images":169733, "num_eval_images":42434, "num_label_classes":100,"data_dir":'gs://ise-bucket/efficientnet/dataset-tf-20-01-13_1-img_10000-ent_100-class'},
    "cifar100":{"num_train_images":50000, "num_eval_images":10000, "num_label_classes":100,"data_dir":"gs://ise-bucket/cifar100"},
    "dataset-tf-balanced":{"num_train_images":159460, "num_eval_images":39866, "num_label_classes":100,"data_dir":'gs://ise-bucket/efficientnet/dataset-tf-20-01-15_1-img_2000-ent_100-class'}
                   }


model_dir = 'gs://ise-bucket/efficientnet/'+model_name+'/'+run_name

init_checkpoint_dir = 'gs://ise-bucket/efficientnet/'+model_name+'/'

model_params = model_params[model_name]
data_set_params = data_set_params[data_set]

use_tpu = True
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
input_image_size = efficientnet_builder.efficientnet_params(model_name)[2] 
data_dir = data_set_params["data_dir"]
iterations_per_loop = model_params["iterations_per_loop"]
num_train_images = data_set_params["num_train_images"]
num_eval_images =  data_set_params["num_eval_images"]
train_batch_size = model_params["train_batch_size"]
eval_batch_size = 1024
predict_batch_size = 1024
log_step_count_steps = 64

transpose_input=False
use_cache = True
num_parallel_calls = 8
num_label_classes = data_set_params["num_label_classes"]
include_background_label = False
augment_name = None #"randaugment" #None #randaugment
mixup_alpha = 0.0
randaug_num_layers = 0 #2 #0 #1-3
randaug_magnitude = 0 #20 #0 #5-30

num_epochs = 100
train_steps = int(num_train_images / train_batch_size * num_epochs)
eval_steps = int(num_eval_images / eval_batch_size)
moving_average_decay = 0.9999
label_smoothing = 0.1
weight_decay = 1e-5
base_learning_rate = 0.016 #0.016 standard 
skip_host_call = False

"""###Helper Functions"""

def get_ema_vars():
  #print("Get all exponential moving average (ema) variables.")
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  #print(list(set(ema_vars)))
  return list(set(ema_vars))

"""###InputFn"""

def input_fn_builder(is_training):
  def input_fn(params):
      """Input function which provides a single batch for train or eval.
      Args:
        params: `dict` of parameters passed from the `TPUEstimator`.
            `params['batch_size']` is always provided and should be used as the
            effective batch size.
      Returns:
        A `tf.data.Dataset` object.
      """

      print(params)

      # Retrieves the batch size for the current shard. The # of shards is
      # computed according to the input pipeline deployment. See
      # tf.estimator.tpu.RunConfig for details.
      batch_size = params['batch_size'] #batch_size = train_batch_size / num_cores

      num_cores = 8
      imagenet_train = imagenet_input.ImageNetInput(
        is_training=is_training,
        data_dir=data_dir,
        transpose_input=transpose_input,
        cache=use_cache and is_training,
        image_size=input_image_size,
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=False,
        num_label_classes=num_label_classes,
        include_background_label=include_background_label,
        augment_name=augment_name,
        mixup_alpha=mixup_alpha,
        randaug_num_layers=randaug_num_layers,
        randaug_magnitude=randaug_magnitude)

      if 'context' in params:
        current_host = params['context'].current_input_fn_deployment()[1]
        num_hosts = params['context'].num_hosts
      else:
        current_host = 0
        num_hosts = 1

      dataset = imagenet_train.make_source_dataset(current_host, num_hosts)
      
      # Use the fused map-and-batch operation.
      #
      # For XLA, we must used fixed shapes. Because we repeat the source training
      # dataset indefinitely, we can use `_remainder=True` to get fixed-size
      # batches without dropping any training examples.
      #
      # When evaluating, `drop_remainder=True` prevents accidentally evaluating
      # the same image twice by dropping the final batch if it is less than a full
      # batch size. As long as this validation is done with consistent batch size,
      # exactly the same images will be used.
      dataset = dataset.apply(
          tf.data.experimental.map_and_batch(
              imagenet_train.dataset_parser, batch_size=batch_size,
              num_parallel_batches=num_cores, drop_remainder=True))

      # Apply Mixup
      if imagenet_train.is_training and imagenet_train.mixup_alpha > 0.0:
        dataset = dataset.map(
            functools.partial(imagenet_train.mixup, batch_size, imagenet_train.mixup_alpha),
            num_parallel_calls=num_cores)

      # Transpose for performance on TPU
      if imagenet_train.transpose_input:
        dataset = dataset.map(
            lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
            num_parallel_calls=num_cores)

      # Assign static batch size dimension
      dataset = dataset.map(functools.partial(imagenet_train.set_shapes, batch_size))

      # Prefetch overlaps in-feed with training
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      
      #print("transformed data:",dataset)

      return dataset
  return input_fn

"""###ModelFn"""

def model_fn(features, labels, mode, params):
  """The model_fn to be used with TPUEstimator.
  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of one hot labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.
  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  stats_shape = [1, 1, 3]

  if transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (moving_average_decay > 0)

  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)
  tf.logging.info('Using open-source implementation.')
  override_params = {}

  def normalize_features(features, mean_rgb, stddev_rgb):
    """Normalize the image given the means and stddevs."""
    features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
    features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
    return features

  def build_model():
    """Build model using the model_name given through the command line."""
    model_builder = efficientnet_builder
    normalized_features = normalize_features(features, model_builder.MEAN_RGB,
                                             model_builder.STDDEV_RGB)
    output_layer, _ = model_builder.build_model(
        normalized_features,
        model_name=model_name,
        fine_tuning = False,
        pooled_features_only = True,
        features_only = False,
        training=is_training,
        override_params=override_params,
        model_dir=model_dir) #model_dir used for saving configs
    
    hidden_size = output_layer.shape[-1].value

    num_labels = num_label_classes
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)

    logits = tf.nn.bias_add(logits, output_bias)
  
    return logits

  logits = build_model()

  train_op = None
  host_call = None


  # PREDICTION MODE
  # ******************************************************************************
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'label': tf.argmax(labels, axis=1),
        'prediction': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })


  # CONTINUE WITH TRAINING OR EVAL MODE
  # ******************************************************************************
  batch_size = params['batch_size']

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=labels,
      label_smoothing=label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  global_step = tf.train.get_global_step()
  tf.logging.info("globalstep:{}".format(global_step))

  restore_vars_dict = None
  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=global_step)
    ema_vars = get_ema_vars()
    restore_vars_dict = ema.variables_to_restore(ema_vars)
    #print("restore_vars_dict")
    #print(restore_vars_dict)

  
  # TRAINING MODE
  # ******************************************************************************
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (tf.cast(global_step, tf.float32) / params['steps_per_epoch'])
    tf.logging.info('current epoch = {}'.format(tf.reshape(current_epoch, [1])))
    
    scaled_lr = base_learning_rate * (train_batch_size / 256.0)
    tf.logging.info('base_learning_rate = %f', base_learning_rate)
    learning_rate = utils.build_learning_rate(scaled_lr, global_step,
                                             params['steps_per_epoch'])
    
    # Override scaled learning rate with base_learning_rate (comment if scaled lr should be used)
    learning_rate = base_learning_rate #new


    optimizer = utils.build_optimizer(learning_rate)
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    # Host Call Function to write scalar summaries for Tensorboard
    def host_call_fn(gs, lr, ce,acc):
      print("host call")
      gs = gs[0]

      with tf2.summary.create_file_writer(model_dir, max_queue=iterations_per_loop).as_default():
        with tf2.summary.record_if(True):
          tf2.summary.scalar('learning_rate', lr[0], step=gs)
          tf2.summary.scalar('current_epoch', ce[0], step=gs)
          tf2.summary.scalar('accuracy', acc[0], step=gs)
        return tf.summary.all_v2_summary_ops()

    gs_t = tf.reshape(global_step, [1])
    lr_t = tf.reshape(learning_rate, [1])
    ce_t = tf.reshape(current_epoch, [1])

    labels = tf.argmax(labels, axis=1)
    predictions = tf.argmax(logits, axis=1)
    #accuracy = tf.metrics.accuracy(labels, predictions)  # tf running accuracy metric (jumps after restart)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)) # own accuracy metric
    accuracy_t = tf.reshape(accuracy, [-1])
    host_call = (host_call_fn, [gs_t, lr_t, ce_t,accuracy_t])



  # EVALUATION MODE
  # ******************************************************************************
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    # Function to compute confusion matrix with labels and predictions
    def eval_confusion_matrix(labels, predictions):
        with tf.variable_scope("eval_confusion_matrix"):
          con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_label_classes)

          con_matrix_sum = tf.Variable(lambda: tf.zeros(shape=(num_label_classes,num_label_classes), dtype=tf.int32),
                                              trainable=False,
                                              name="confusion_matrix_result",
                                              collections=[tf.GraphKeys.LOCAL_VARIABLES])


          update_op = tf.assign_add(con_matrix_sum, con_matrix)

          return tf.convert_to_tensor(con_matrix_sum), update_op

    # Metric Function to return and write evaluation metrics 
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.
      Args:
        labels: `Tensor` with shape `[batch, num_classes]`.
        logits: `Tensor` with shape `[batch, num_classes]`.
      Returns:
        A dict of the metrics to return from evaluation.
      """
      labels = tf.argmax(labels, axis=1)
      predictions = tf.argmax(logits, axis=1)

      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)
      con_mat= eval_confusion_matrix(labels, predictions)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
          'con_mat':con_mat
      }

    eval_metrics = (metric_fn, [labels, logits])



  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: %d', num_params)

  tvars = tf.trainable_variables()
  
  if init_checkpoint_dir:

    # TPU Scaffold Function to load variables from initial pretrained checkpoint 
    def tpu_scaffold():
      print("init_checkpoint_dir: {}".format(init_checkpoint_dir))

      # Remove variables that are not stored in pretrained checkpoint
      restore_vars_dict.pop("global_step")
      restore_vars_dict.pop("output_bias/ExponentialMovingAverage")
      restore_vars_dict.pop("output_weights/ExponentialMovingAverage")
      tf.train.init_from_checkpoint(init_checkpoint_dir, restore_vars_dict)
      return tf.train.Scaffold()

    scaffold_fn = tpu_scaffold

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)

"""###Run Config and Estimator"""

params = dict(
      steps_per_epoch=num_train_images / train_batch_size,
      use_bfloat16=False);

save_checkpoints_steps = 5*params["steps_per_epoch"];

config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=model_dir, #model dir used for 
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max = 100,
      log_step_count_steps=log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2));

est = tf.estimator.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=config,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size = 1024,
      params=params);

"""### Train"""

def training(epochs):
  start_timestamp = datetime.datetime.now() # This time will include compilation time
  tf.logging.info("*"*100)
  tf.logging.info("Start training for {} epochs at {}".format(epoch, start_timestamp))
  current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)
  steps_this_run = np.min([epochs*params['steps_per_epoch'],train_steps-current_step]) # runs for epochs minimum per run and maximum to train_steps
  tf.logging.info(
      'Training for %d steps (%.2f epochs in total). Current'
      ' step %d.', steps_this_run,
      steps_this_run / params['steps_per_epoch'], current_step)
  tf.contrib.summary.always_record_summaries()
  est.train(
      input_fn=input_fn_builder(is_training=True),
      max_steps=int(current_step+steps_this_run))

  end_timestamp = datetime.datetime.now() # This time will include compilation time
  tf.logging.info("Finished training at {}".format(end_timestamp))
  tf.logging.info("*"*100)

training(1)

"""###Evaluation"""

def evaluation():
  start_timestamp = datetime.datetime.now() # This time will include compilation time
  tf.logging.info("*"*100)
  current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)
  epoch = current_step / params['steps_per_epoch']
  tf.logging.info("Start evaluation epoch {} at {}".format(epoch,start_timestamp))
  predictions = est.evaluate(input_fn = input_fn_builder(is_training=False), steps = eval_steps)
  end_timestamp = datetime.datetime.now() # This time will include compilation time
  tf.logging.info("Finished evaluation at {}".format(end_timestamp))
  tf.logging.info("*"*100)
  return predictions

evaluation(1)

"""Train and Evaluation Loop"""

# Train and Evaluation Loop for Test Purposes
print(train_steps)

for i in range(3):
  training(i+1)

  try:  
    ckpt_step_after_training = int(tf.train.latest_checkpoint(model_dir).replace(model_dir+"/model.ckpt-",""))
  except:
    print("df")
    break;

  evaluation(i+1)

"""#Evaluation"""

dfdd = est.predict(input_fn=input_fn_builder(is_training=False),yield_single_examples=True)
x = list(dfdd)

"""store predictions to csv files"""

def store_predictions_to_csv(predict_result):
  list_of_probs = []
  list_of_labels = []

  for image in predict_result:
    list_of_probs.append(image["probabilities"])
    list_of_labels.append(image["label"])

  pd.DataFrame(list_of_probs).to_csv("confidences.csv")
  pd.DataFrame(list_of_labels).to_csv("labels.csv")

store_predictions_to_csv(x)

df = pd.read_csv('gs://ise-bucket/efficientnet/configs/config_3/labels_map.csv',index_col=None)
labels_map = dict(zip(df.label, df.label_index))

df = pd.read_csv('gs://ise-bucket/efficientnet/hierarchy_lists/dbo classes hierarchies.csv',index_col=None)
hierarchies_raw = df.values.tolist()
hierarchies = [[i for i in line if type(i) != float] for line in hierarchies_raw]
hierarchies = [[i for i in line if i != "owl:Thing"] for line in hierarchies]
hierarchies = [line for line in hierarchies if line[-1] in labels_map.keys()]
[line.insert(0,labels_map[line[-1]]) for line in hierarchies]
hierarchy_list = hierarchies

#hierarchy_list.append([29,"Disease"])

hierarchy_dict = {}
for li in hierarchy_list:
  hierarchy_dict[li[0]-1] = li[1:]

labels_per_level = {1:[],2:[],3:[],4:[],5:[]}
for li in hierarchy_list:
  level = len(li)-1
  ent = li[-1]
  labels_per_level[level].append(ent)

child_parent_dict = {}
for li in hierarchy_list:
  child = li[-1]
  parent = li[-2]
  if type(parent) != int:
    child_parent_dict[child] = parent

inverse_label_map = {value-1:key for key,value in labels_map.items()}

lvl1_label_to_index = {label:index for index,label in enumerate(labels_per_level[1])}

level_of_label = {v[-1]:len(v) for k,v in hierarchy_dict.items()}

"""calculate cumulated confidences"""

agg_conf_per_image = {}
label_per_image = {}

levels_desc = list(labels_per_level.keys())
levels_desc.sort(reverse=True)
for ent_index,ent in enumerate(x):
  label_conf = {}
  label_cum_conf = {}

  for level in levels_desc:
    labels = labels_per_level[level]
    for label in labels:
        index = labels_map[label]-1
        if label in label_conf.keys():
            label_conf[label] += ent["probabilities"][index]
        else:
            label_conf[label] = ent["probabilities"][index]

        if label in child_parent_dict.keys():
          parent_label = child_parent_dict[label]
          if parent_label in label_conf.keys():
            label_conf[parent_label] += label_conf[label]
          else:
            label_conf[parent_label] = label_conf[label]

  conf = sorted(label_conf.items() , reverse=True, key=lambda x: x[1])
  label_per_image[ent_index] = ent["label"]
  agg_conf_per_image[ent_index] = conf

"""calculate accuracy for level 1 predictions"""

accs_lvl1 = []
rel_data_lvl1 = []
conf_mats_lvl1 = {}
avg_lvl1 = []
levels = []

for conf_threshold in range(0,100,5):  
    conf_threshold = conf_threshold/100
    print("threshold:",conf_threshold)
    scores_lvl1 = 0
    tries = 0
    for ent_index,ent in enumerate(x):
        hierarchy = hierarchy_dict[ent["label"]]
        hierarchy_top_label = hierarchy[0]
        top_pred = agg_conf_per_image[ent_index][0]
        top_pred_label = top_pred[0]
        top_pred_conf = top_pred[1]

        if top_pred_conf >= conf_threshold:
            tries += 1
            if top_pred_label == hierarchy_top_label:
                scores_lvl1 += 1
                levels.append(1)
        
    print("acc:",scores_lvl1/tries)
    accs_lvl1.append(scores_lvl1/tries)
    print("% of data predicted:",tries/len(x))
    rel_data_lvl1.append(tries/len(x))
    avg_lvl1.append(np.mean(levels))
    print("avg predicted level:", np.mean(levels))
    print()

"""calculate accuracy for lowest level above threshold"""

accs_lvl_all = []
rel_data_lvl_all = []
conf_mats_lvl_all = {}
avg_lvl_all = []
t = []

for conf_threshold in range(0,100,5):  
    conf_threshold = conf_threshold/100
    print("threshold:",conf_threshold)
    scores_lvl_all = 0
    tries = 0
    levels_of_pred = []

    for ent_index,ent in enumerate(x):
        hierarchy = hierarchy_dict[ent["label"]]
        gt_lvl = level_of_label[inverse_label_map[ent["label"]]]

        levels = list(range(1,5+1))
        levels.sort(reverse=True)

        for lvl in levels:
            labels = labels_per_level[lvl]
            agg_confs = [x for x in agg_conf_per_image[ent_index] if x[0] in labels]
            top_pred = agg_confs[0]
            top_pred_label = top_pred[0]
            top_pred_conf = top_pred[1]

            if top_pred_conf >= conf_threshold:
                tries += 1
                if top_pred_label in hierarchy:
                    scores_lvl_all += 1
                    levels_of_pred.append(lvl)
                    t.append(gt_lvl)
                break;     


    print("acc:",scores_lvl_all/tries)
    accs_lvl_all.append(scores_lvl_all/tries)
    print("% of data predicted:",tries/len(x))
    rel_data_lvl_all.append(tries/len(x))
    avg_lvl_all.append(np.mean(levels_of_pred))
    print("avg predicted level:", np.mean(levels_of_pred))
    print(np.mean(t))
    print()

"""calculate accuracy for predictions on all levels"""

accs = []
rel_data = []
conf_mats = {}
avg_lvl = []
for conf_threshold in range(0,100,5):  
  conf_mat = np.zeros((len(labels_map.keys()),len(labels_map.keys())))
  conf_threshold = conf_threshold/100
  print("threshold:",conf_threshold)
  scores = 0
  tries = 0
  levels_of_pred = []

  for ent_index,ent in enumerate(x):
    gt_index = ent["label"]
    gt_label = inverse_label_map[gt_index]
    pred_index = ent["prediction"]
    pred_label = inverse_label_map[pred_index]
    pred_conf = ent["probabilities"][pred_index]

    if pred_conf >= conf_threshold:
      tries += 1
      conf_mat[gt_index][pred_index] += 1
      if gt_label == pred_label:
        scores += 1
        levels_of_pred.append(level_of_label[pred_label])

  conf_mats[conf_threshold]=conf_mat      

  print("acc:",scores/tries)
  accs.append(scores/tries)
  print("% of data predicted:",tries/len(x))
  avg_lvl.append(np.mean(levels_of_pred))
  print("avg predicted level:", np.mean(levels_of_pred))
  print()
  rel_data.append(tries/len(x))

"""plot accuracies"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(20,10)) 
plt.xlabel('% of data predicted')
plt.ylabel('prediction accuracy')     
annotations = [x/100 for x in list(range(0,100,5))]
ax.plot(rel_data,accs,".-r", label="standard classifier predictions above threshold")
ax.plot(rel_data_lvl_all,accs_lvl_all,".-g",label="predictions on lowest possible level above threshold")
ax.plot(rel_data_lvl1,accs_lvl1,".-b",label="predictions on top entity level above threshold")

ax2 = ax.twinx()
ax2.set_ylabel("average level predicted")
ax2.plot(rel_data,avg_lvl,":r")
ax2.plot(rel_data_lvl1,avg_lvl1,":b")
ax2.plot(rel_data_lvl_all,avg_lvl_all,":g")


ax.set_xlim(1.05, -0.05) 
plt.xticks(np.arange(0, 1.1, step=0.1))
extraString = 'confidence threshold'
handles, labels = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', label=extraString))
plt.legend(handles=handles,loc="lower right")
for index,annotation in enumerate(annotations):
  if (index+1) % 2 == 0:
    ax.text(rel_data[index]+0.01, accs[index]+0.005,annotation,va="bottom")
    ax.text(rel_data_lvl1[index]+0.01, accs_lvl1[index]+0.005,annotation,va="bottom")
    ax.text(rel_data_lvl_all[index]+0.01, accs_lvl_all[index]+0.005,annotation,va="bottom")

"""####Predict single Image"""

class EvalCkptDriver(ut.EvalCkptDriver):
  """A driver for running eval inference."""

  def build_model(self, features, is_training):
    """Build model using the model_name given through the command line."""
    model_builder = efficientnet_builder

    
    features -= tf.constant(
        model_builder.MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(
        model_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
    

    output_layer, _ = model_builder.build_model(
        features,
        model_name=model_name,
        fine_tuning = False,
        pooled_features_only = True,
        features_only = False,
        training=is_training,
        model_dir=model_dir)#model_dir used for saving configs
    
    hidden_size = output_layer.shape[-1].value

    num_labels = num_label_classes
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)

    logits = tf.nn.bias_add(logits, output_bias)

    probs = tf.nn.softmax(logits)
    probs = tf.squeeze(probs)
    return probs

  def get_preprocess_fn(self):
    """Build input dataset."""
    return preprocessing.preprocess_image

  def eval_example_images(self,
                          ckpt_dir,
                          image_files,
                          labels_map_file,
                          enable_ema=True,
                          export_ckpt=None):
    """Eval a list of example images.

    Args:
      ckpt_dir: str. Checkpoint directory path.
      image_files: List[str]. A list of image file paths.
      labels_map_file: str. The labels map file path.
      enable_ema: enable expotential moving average.
      export_ckpt: export ckpt folder.

    Returns:
      A tuple (pred_idx, and pred_prob), where pred_idx is the top 5 prediction
      index and pred_prob is the top 5 prediction probability.
    """
    classes = json.loads(tf.gfile.Open(labels_map_file).read())
    out_probs_list, pred_idx, pred_prob = self.run_inference(
        ckpt_dir, image_files, [0] * len(image_files), enable_ema, export_ckpt)
    labels_list = []
    for i in range(len(image_files)):
      print('predicted class for image {}: '.format(image_files[i]))
      ground_truth = image_files[i].replace("gs://ise-bucket/efficientnet/dataset-20-01-13_1-img_10000-ent_100-class/","").split("_")[0]
      print("ground truth: {}".format(ground_truth))
      labels_list.append(ground_truth)
      for j, idx in enumerate(pred_idx[i]):
        print('  -> top_{} ({:4.2f}%): {}  '.format(j, pred_prob[i][j] * 100,
                                                    classes[str(idx)]))
    return labels_list,out_probs_list, pred_idx, pred_prob

  def run_inference(self,
                    ckpt_dir,
                    image_files,
                    labels,
                    enable_ema=True,
                    export_ckpt=None):
    """Build and run inference on the target images and labels."""
    label_offset = 1 if self.include_background_label else 0
    with tf.Graph().as_default(), tf.Session() as sess:
      images, labels = self.build_dataset(image_files, labels, False)
      probs = self.build_model(images, is_training=False)
      if isinstance(probs, tuple):
        probs = probs[0]

      self.restore_model(sess, ckpt_dir, enable_ema, export_ckpt)
      tf.logging.info("model restored")
      prediction_idx = []
      prediction_prob = []
      out_probs_list = []
      for i in range(len(image_files) // self.batch_size):
        out_probs = sess.run(probs)
        idx = np.argsort(out_probs)[::-1]
        prediction_idx.append(idx[:5] - label_offset)
        prediction_prob.append([out_probs[pid] for pid in idx[:5]])
        out_probs_list.append(out_probs)
        print("predicted image nr",i)

      # Return the top 5 predictions (idx and prob) for each image.
      return out_probs_list, prediction_idx, prediction_prob



def get_eval_driver(model_name,image_size,num_classes,include_background_label=False):
  return EvalCkptDriver(
      model_name=model_name,
      batch_size=1,
      image_size=input_image_size,
      num_classes = num_label_classes,
      include_background_label=include_background_label)

"""predict single image"""

import tensorflow.compat.v1 as tf
import utils as ut
import preprocessing
from IPython import display

!wget https://www.travelbook.de/data/uploads/2017/09/gettyimages-463523885_1506503445-1040x690.jpg -O horst.jpeg
image_file = 'horst.jpeg'
display.display(display.Image(image_file))

if not os.path.exists('label_map.txt'):
  with open('label_map.txt', 'w') as file:
      file.write(json.dumps(inverse_label_map))

labels_map_file = "label_map.txt"
image_files = [image_file]

eval_driver = get_eval_driver(
      model_name=model_name,
      image_size=input_image_size,
      num_classes = num_label_classes,
      include_background_label=include_background_label)


labels_list, out_probs, pred_idx, pred_prob = eval_driver.eval_example_images(model_dir, image_files, labels_map_file)

"""predict multiple images"""

df_train = pd.read_csv("gs://ise-bucket/efficientnet/configs/config_3/validation_path_label.csv",header=0)
validation_files = list(df_train["url"])
 
  
eval_driver = get_eval_driver(
      model_name=model_name,
      image_size=input_image_size,
      num_classes = num_label_classes,
      include_background_label=include_background_label)


labels_list, out_probs, pred_idx, pred_prob = eval_driver.eval_example_images(model_dir,  validation_files[0:1000], labels_map_file)

agg_conf_per_image = {}
label_per_image = {}

levels_desc = list(labels_per_level.keys())
levels_desc.sort(reverse=True)
for ent_index,ent in enumerate(zip(labels_list,out_probs)):
  label_conf = {}
  label_cum_conf = {}

  for level in levels_desc:
    labels = labels_per_level[level]
    for label in labels:
        index = labels_map[label]-1
        if label in label_conf.keys():
            label_conf[label] += ent[1][index]
        else:
            label_conf[label] = ent[1][index]

        if label in child_parent_dict.keys():
          parent_label = child_parent_dict[label]
          if parent_label in label_conf.keys():
            label_conf[parent_label] += label_conf[label]
          else:
            label_conf[parent_label] = label_conf[label]

  conf = sorted(label_conf.items() , reverse=True, key=lambda x: x[1])
  label_per_image[ent_index] = ent[0]
  agg_conf_per_image[ent_index] = conf

for conf_threshold in range(0,100,5):  
  conf_threshold = conf_threshold/100
  print("threshold",conf_threshold)
  predictions = []
  for idx in range(0,len(out_probs)):
    top_pred = get_prediction_for_threshold(conf_threshold,agg_conf_per_image[idx])
    predictions.append(top_pred)


  scores = 0
  tries = 0
  for tup in zip(labels_list,predictions):
    if tup[1] is not None:
      label = tup[0]
      pred = tup[1]
      hierarchy = hierarchy_dict[labels_map[label]-1]
      tries += 1
      if pred in hierarchy:
        scores += 1
                      
  print("acc:",scores/tries)
  print("% of data predicted:",tries/len(labels_list))
  print("\n\n")

def get_prediction_for_threshold(conf_threshold, agg_confs):
  levels = list(range(1,5+1))
  levels.sort(reverse=True)
  #print("threshold:",conf_threshold)

  for lvl in levels:
    labels = labels_per_level[lvl]
    filtered_agg_confs = [x for x in agg_confs if x[0] in labels]
    top_pred = filtered_agg_confs[0]
    top_pred_label = top_pred[0]
    top_pred_conf = top_pred[1]

    if top_pred_conf >= conf_threshold:
      #print(top_pred_label)
      return top_pred_label
      break;


for conf_threshold in range(0,100,5):  
  conf_threshold = conf_threshold/100
  get_prediction_for_threshold(conf_threshold,agg_conf)

"""create and save confusion_matrix"""

import seaborn as sns
is_toplvl = True
threshold = 0
fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
if is_toplvl: 
  ax = sns.heatmap(conf_mats_lvl1[threshold], xticklabels=lvl1_label_to_index.keys(),yticklabels=lvl1_label_to_index.keys(),cmap=cmap)
else:
  ax = sns.heatmap(conf_mats_lvl1[threshold], xticklabels=labels_map.keys(),yticklabels=labels_map.keys(),cmap=cmap)
plt.savefig("output.png")

"""#Resteverwertung

##Testecke
"""

print(model_dir+"/checkpoint")
c = tf.gfile.GFile(model_dir+"/checkpoint")
for line in c.readlines():
  print(line)

"""##Save finetuning files"""

credentials, project = google.auth.default()
client = storage.Client(credentials=credentials, project='ise-project-259623')
bucket = client.get_bucket('ise-bucket')

def save_finetuning(foldername):
  bucket = client.get_bucket("ise-bucket")

  blobs = tf.gfile.ListDirectory(model_dir)
  for blob_name in blobs:
    blob = bucket.get_blob('efficientnet/'+model_name+'/FT/'+blob_name)
    if blob != None:
      new_name = 'efficientnet/'+model_name+'/'+foldername+'/'+blob_name
      bucket.rename_blob(blob,new_name)

  blobs = tf.gfile.ListDirectory(model_dir+"/eval")
  for blob_name in blobs:
    blob = bucket.get_blob('efficientnet/'+model_name+'/FT/eval/'+blob_name)
    new_name = 'efficientnet/'+model_name+'/'+foldername+'/eval/'+blob_name
    bucket.rename_blob(blob,new_name)

save_finetuning("FT-13-01-20-1")

!gcloud auth application-default login

credentials, project = google.auth.default()
client = storage.Client(credentials=credentials, project='ise-project-259623')
bucket = client.get_bucket('ise-bucket')

bucket = client.get_bucket("ise-bucket")

blobs = tf.gfile.ListDirectory(data_dir+"/")
for blob_name in blobs:
  if blob_name.startswith("train"):
    blob = bucket.get_blob('efficientnet/dataset_tf_large/'+blob_name)
    print(blob)
    if blob != None:
      if blob.
    #  new_name = 'efficientnet/dataset_tf_large/'+blob_name
    #  bucket.rename_blob(blob,new_name)

"""#### Delete GCS Folder"""

def delete_folder_from_gcs(folder):
  print(folder)
  blobs = bucket.list_blobs(prefix = folder)
  l = [blob for blob in blobs]
  print(len(l))
  index = 1
  for blob in l:
    try:
      blob.delete()
      index += 1
      if index % 100 == 0:
        print("delete image",index,"from",len(l),":",blob.name)
    except Exception as e:
      continue

delete_folder_from_gcs("efficientnet/dataset2/")

"""create hierarchy from dbpedia.owl file"""

credentials, project = google.auth.default()
client = storage.Client(credentials=credentials, project='ise-project-259623')
bucket = client.get_bucket('ise-bucket')

df = pd.read_csv('gs://ise-bucket/efficientnet/configs/config_3/labels_map.csv',index_col=None)
labels_map = dict(zip(df.label, df.label_index))

#@title
#extract list with (parent,child) tuples from dbpedia.owl file
xmlblob = bucket.blob("efficientnet/configs/config_1/dbpedia.owl")
inputxml = xmlblob.download_as_string()

tree = ET.parse(BytesIO(inputxml))
root = tree.getroot()

parent_child_list = []

for elem in iter(root.findall("./{http://www.w3.org/2002/07/owl#}Class")):
  owl_class = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about').replace("http://dbpedia.org/ontology/","")
  owl_parent_class = elem.find("./{http://www.w3.org/2000/01/rdf-schema#}subClassOf").get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource').replace("http://dbpedia.org/ontology/","")
  if not owl_parent_class == "http://www.w3.org/2002/07/owl#Thing":
    parent_child_list.append((owl_parent_class,owl_class))
    

#create tree with dictionaries {parent1: {child11: {child111:{...}, child112:{...}}, child12}, parent2:{...}}} from parent_child_list
def get_children(parent, relations):
    children = (r[1] for r in relations if r[0] == parent)
    return {c: get_children(c, relations) for c in children}

parents, children = map(set, zip(*parent_child_list))
the_tree = {p: get_children(p, parent_child_list) for p in (parents - children)}


#create list with hierarchy in sequential form: [[Agent, Person, Athlete, GridironFootballPlayer, AmericanFootballPlayer],[Agent, Person, Athlete],[...],...]
hierarchy_list = []

def create_hierarchy_list(parent, i ,element):
  for k in element.keys(): #search for children in element
    classes_so_far = parent+" "+k #append to string with all higher nodes than children k
    
    #if children k is not used in datasets: break and remove it from string with all higher nodes than children k
    if not k in labels_map.keys():
      classes_so_far = " ".join(classes_so_far.split(" ")[:-1])
    #if children k is used: save it to hierachy_list
    else:
      li = classes_so_far.split(" ")[1:]
      li.insert(0,labels_map[k])
      hierarchy_list.append(li)

    #search recursively
    create_hierarchy_list(classes_so_far,i+1,element[k]);
  
create_hierarchy_list("",0,the_tree)

df = pd.DataFrame(hierarchy_list,columns=["label_index","level1","level2","level3","level4","level5"]).set_index("label_index").sort_index()