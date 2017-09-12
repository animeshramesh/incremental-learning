import tensorflow as tf
import os, re

def get_checkpoints(checkpoint_dir):
    '''
    Finds all checkpoints in a directory and returns them in order
    from least iterations to most iterations
    '''
    meta_list=[]
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.meta'):
            meta_list.append(os.path.join(checkpoint_dir, file[:-5]))
    meta_list = sort_nicely(meta_list)
    return meta_list

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    From Ned Batchelder
    https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l

def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def optimistic_restore(session, save_file, variable_scope=''):
    '''
    A Caffe-style restore that loads in variables
    if they exist in both the checkpoint file and the current graph.
    Call this after running the global init op.
    By DanielGordon10 on December 27, 2016
    https://github.com/tensorflow/tensorflow/issues/312
    With RalphMao tweak.

    bpugh, July 21, 2017: Added a variable_scope so that a network can be
    loaded within a tf.variable_scope() and still have weights restored.
    '''
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    if variable_scope is '':
        saved_shapes_scoped = saved_shapes
        offset = 0
    else:
        saved_shapes_scoped = [variable_scope + '/' + x for x in saved_shapes]
        offset = len(variable_scope) + 1

    var_names = []
    for var in tf.global_variables():
        search_term = var.name.split(':')[0]
        if search_term in saved_shapes_scoped:
            var_names.append((var.name.split(':')[0], var.name.split(':')[0][offset:]))

    name2var = dict(zip(map(lambda x:x.name.split(':')[0],
            tf.global_variables()), tf.global_variables()))
    restore_variables = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = name2var[var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    found_variable = tf.get_variable(var_name)
                    restore_variables.append(found_variable.assign(reader.get_tensor(saved_var_name)))
            except:
                print("{} couldn't be loaded.".format(saved_var_name))
    session.run(restore_variables)
