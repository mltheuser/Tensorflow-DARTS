from abc import abstractmethod
from typing import Callable, List

import tensorflow as tf


class ArchitectureSearchLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._trainable_architecture = tf.Variable(True)

    @property
    @abstractmethod
    def trainable_architecture_variables(self):
        pass

    @property
    @abstractmethod
    def trainable_operational_variables(self):
        pass

    @property
    def trainable_architecture(self):
        return tf.constant(self._trainable_architecture)

    @trainable_architecture.setter
    def trainable_architecture(self, value: bool):
        self._trainable_architecture.assign(value)


class ArchitectureSearchModel(tf.keras.Model, ArchitectureSearchLayer):
    @property
    def trainable_architecture_variables(self):
        tav = []
        for layer in self.layers:
            if isinstance(layer, ArchitectureSearchLayer):
                tav += layer.trainable_architecture_variables
        return tav

    @property
    def trainable_operational_variables(self):
        tav = self.trainable_architecture_variables
        tov = self.trainable_variables
        for var in tav:
            tov.remove(var)
        return tov

    @property
    def trainable_architecture(self):
        return tf.constant(self._trainable_architecture)

    @trainable_architecture.setter
    def trainable_architecture(self, value: bool):
        self._trainable_architecture.assign(value)
        for layer in self.layers:
            if isinstance(layer, ArchitectureSearchLayer):
                layer.trainable_architecture = value


class MixedOp(ArchitectureSearchLayer):
    def __init__(self, operations: List[tf.keras.layers.Layer], num_on_samples: int):
        super(MixedOp, self).__init__()
        self._num_on_samples = num_on_samples

        self._ops = operations
        self._num_ops = len(self._ops)
        self._logits = tf.Variable(1e-3 * tf.random.normal(shape=(self._num_ops,)))

        assert self._num_ops >= self._num_on_samples

    def __call__(self, x, training=None, *args, **kwargs):
        return tf.cond(
            tf.logical_and(self._trainable_architecture, training),
            true_fn=lambda: self.stochastic_call(x, training),
            false_fn=lambda: self.non_stochastic_call(x, training)
        )

    @abstractmethod
    def stochastic_call(self, x, training, *args, **kwargs):
        pass

    def call_op(self, data):
        op_id, x, training = data
        op_results = [lambda: op(x, training=training) for op in self._ops]
        return tf.switch_case(branch_index=op_id, branch_fns=op_results)

    def evaluate_operations_partially(self, x, training, op_ids):
        if tf.executing_eagerly():
            return tf.stack([self._ops[op_id](x, training=training) for op_id in op_ids])
        else:
            expanded_x = tf.repeat(tf.expand_dims(x, axis=0), repeats=self._num_on_samples, axis=0)
            t_training = tf.repeat(tf.expand_dims(training, axis=0), repeats=self._num_on_samples, axis=0)
            return tf.map_fn(fn=self.call_op, elems=(op_ids, expanded_x, t_training),
                             fn_output_signature=tf.float32)

    def non_stochastic_call(self, x, training):
        top_k_ids = tf.math.top_k(self._logits, k=self._num_on_samples).indices

        op_results = self.evaluate_operations_partially(x, training, top_k_ids)

        result = tf.reduce_sum(op_results, axis=0)
        return result

    @property
    def trainable_architecture_variables(self):
        return [self._logits]

    @property
    def trainable_operational_variables(self):
        tov = self.trainable_variables
        tov.remove(self._logits)
        return tov


class ContinuousMixedOp(MixedOp):
    def stochastic_call(self, x, training, *args, **kwargs):
        weights = tf.math.softmax(self._logits, axis=-1)

        op_results = tf.stack([op(x, training=training) for op in self._ops])

        op_results_shape = tf.shape(op_results)
        shape_extension = tf.cast(op_results_shape[1:] / op_results_shape[1:], dtype=tf.int32)
        broadcast_target = tf.concat([tf.expand_dims(op_results_shape[0], axis=0), shape_extension], axis=0)
        broadcasted_weights = tf.reshape(weights, broadcast_target)

        result = tf.reduce_sum(broadcasted_weights * op_results, axis=0)
        return result


def sample_without_replacement(logits, num_samples):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices


def sample_remaining(mask, logits, remaining_samples):
    # remove already sampled on indices
    inverse_mask = tf.logical_not(tf.cast(mask, dtype=tf.bool))
    masked_logits = tf.boolean_mask(logits, inverse_mask)
    masked_logits_shape = tf.shape(masked_logits)
    sampled_indices_eval = sample_without_replacement(tf.expand_dims(masked_logits, axis=0), remaining_samples)[0]
    # adjust sampled indices
    eval_mask = tf.reduce_sum(tf.one_hot(sampled_indices_eval, depth=masked_logits_shape[0]), axis=0)
    eval_mask_adjusted = tf.scatter_nd(tf.cast(tf.where(inverse_mask), dtype=tf.int32), eval_mask, shape=tf.shape(mask))

    eval_mask_full = mask + eval_mask_adjusted
    sorted_sampled_indices_eval = tf.reshape(tf.cast(tf.where(eval_mask_full), dtype=tf.int32), shape=(-1,))
    return sorted_sampled_indices_eval


@tf.custom_gradient
def st_onehot_categorical(weights, num_eval_samples, num_on_samples=1):
    def backward(dsampled_indices_eval, sampled_indices_on, dweights):
        return dweights, None, None

    logits = tf.math.log(weights)
    logits_expanded = tf.expand_dims(logits, axis=0)
    sampled_indices_on = sample_without_replacement(logits_expanded, num_on_samples)[0]
    mask = tf.reduce_sum(tf.one_hot(sampled_indices_on, depth=logits.shape[0]), axis=0)
    sorted_sampled_indices_on = tf.reshape(tf.cast(tf.where(mask), dtype=tf.int32), shape=(-1,))
    remaining_samples = num_eval_samples - num_on_samples

    logits_shape = tf.shape(logits)
    sorted_sampled_indices_eval = tf.cond(
        num_eval_samples >= logits_shape[0],
        true_fn=lambda: tf.range(logits_shape[0], dtype=tf.int32),
        false_fn=lambda: tf.cond(
            tf.logical_and(num_eval_samples > 1, remaining_samples > 0),
            true_fn=lambda: sample_remaining(mask, logits, remaining_samples),
            false_fn=lambda: sorted_sampled_indices_on,
        )
    )

    return (sorted_sampled_indices_eval, sorted_sampled_indices_on, mask), backward


class PartiallyEvaluatedMixOp(MixedOp):
    def __init__(self, operations: List[Callable], num_eval_samples: int, num_on_samples: int):
        super().__init__(operations, num_on_samples)
        self._num_eval_samples = num_eval_samples

        assert self._num_on_samples <= self._num_eval_samples

    @abstractmethod
    def stochastic_call(self, x, training, *args, **kwargs):
        pass


class BinaryMixedOp(MixedOp):
    def stochastic_call(self, x, training, *args, **kwargs):
        return self.call_internal(x, training=training, num_eval_samples=self._num_ops,
                                  num_on_samples=self._num_on_samples)

    def call_internal(self, x, training, num_eval_samples, num_on_samples):
        weights = tf.math.softmax(self._logits, axis=-1)
        sampled_indices_eval, sampled_indices_on, mask = st_onehot_categorical(weights, num_eval_samples,
                                                                               num_on_samples)

        # this works because sampled_indices_on and sampled_indices_eval are always sorted in ascending order.
        op_results = self.evaluate_operations_partially(x, training, sampled_indices_on)

        op_result = self.ops_times_mask_binary(x, training, op_results, mask, sampled_indices_on, sampled_indices_eval)
        return op_result

    def evaluate_remaining(self, x, training, not_evaluated_indices, op_results):
        new_op_results = self.evaluate_operations_partially(x, training, not_evaluated_indices)
        complete_op_results = tf.concat([op_results, new_op_results], axis=0)

        return complete_op_results

    @tf.custom_gradient
    def ops_times_mask_binary(self, x, training, op_results, mask, sampled_indices_on, sampled_indices_eval):
        def backward(dop_result):
            mask_shape = tf.shape(mask)
            eval_mask = tf.reduce_sum(tf.one_hot(sampled_indices_eval, depth=mask_shape[0]), axis=0)
            not_evaluated_mask = tf.logical_and(tf.cast(eval_mask, dtype=tf.bool),
                                               tf.logical_not(tf.cast(mask, dtype=tf.bool)))
            not_evaluated_indices = tf.reshape(tf.cast(tf.where(not_evaluated_mask), dtype=tf.int32), shape=(-1,))

            complete_op_results = tf.cond(
                tf.equal(tf.shape(not_evaluated_indices)[0], 0),
                true_fn=lambda: op_results,
                false_fn=lambda: self.evaluate_remaining(x, training, not_evaluated_indices, op_results),
            )

            d_mask_ill_shaped = tf.reduce_sum(tf.reshape(
                tf.multiply(tf.expand_dims(dop_result, axis=0), complete_op_results),
                shape=(tf.shape(complete_op_results)[0], -1)), axis=-1)

            op_results_shape = tf.shape(op_results)
            on_dmask = tf.scatter_nd(tf.expand_dims(sampled_indices_on, axis=-1),
                                     d_mask_ill_shaped[:op_results_shape[0]], shape=mask_shape)

            not_evaluated_dmask = tf.scatter_nd(tf.expand_dims(not_evaluated_indices, axis=-1),
                                                d_mask_ill_shaped[op_results_shape[0]:], shape=mask_shape)

            dmask = on_dmask + not_evaluated_dmask

            dop_results = tf.repeat(tf.expand_dims(dop_result, axis=0), repeats=self._num_on_samples, axis=0)

            return None, None, dop_results, dmask, None, None

        op_result = tf.reduce_sum(op_results, axis=0)
        return op_result, backward


class BinaryMaskedMixedOp(PartiallyEvaluatedMixOp, BinaryMixedOp):
    def stochastic_call(self, x, training, *args, **kwargs):
        return self.call_internal(
            x, training=training, num_eval_samples=self._num_eval_samples, num_on_samples=self._num_on_samples
        )


@tf.function
def moving_average_update_graph(lr, moving_average, op_results, sampled_indices_eval):
    op_results_mean = tf.reduce_mean(op_results, axis=1)

    update = lr * (op_results_mean - tf.gather(moving_average, sampled_indices_eval))
    moving_average = tf.tensor_scatter_nd_add(moving_average, tf.expand_dims(sampled_indices_eval, axis=-1), update)

    local_activation_estimate = tf.tensor_scatter_nd_update(moving_average,
                                                            tf.expand_dims(sampled_indices_eval, axis=-1),
                                                            op_results_mean)
    return moving_average, local_activation_estimate


class BinaryMovingAverageMixedOp(PartiallyEvaluatedMixOp):
    def __init__(self, operations: List[Callable], num_eval_samples: int, num_on_samples: int, learning_rate: float):
        super(BinaryMovingAverageMixedOp, self).__init__(operations, num_eval_samples, num_on_samples)

        self._moving_average = None
        self._lr = learning_rate

    def stochastic_call(self, x, training, *args, **kwargs):
        weights = tf.math.softmax(self._logits, axis=-1)

        sampled_indices_eval, sampled_indices_on, mask = st_onehot_categorical(weights, self._num_eval_samples,
                                                                               self._num_on_samples)

        # this works because sampled_indices_on and sampled_indices_eval are always sorted in ascending order.
        op_results = tf.stack([self._ops[index_on](x, training=training) for index_on in sampled_indices_on])

        op_result = self.ops_times_mask_binary_ma(x, training, op_results, mask, sampled_indices_on,
                                                  sampled_indices_eval)
        return op_result

    @tf.custom_gradient
    def ops_times_mask_binary_ma(self, x, training, op_results, mask, sampled_indices_on, sampled_indices_eval):
        def backward(dop_result):
            mask_shape = tf.shape(mask)
            eval_mask = tf.reduce_sum(tf.one_hot(sampled_indices_eval, depth=mask_shape[0]), axis=0)
            not_evalated_mask = tf.logical_and(tf.cast(eval_mask, dtype=tf.bool),
                                               tf.logical_not(tf.cast(mask, dtype=tf.bool)))
            not_evalated_indices = tf.cast(tf.reshape(tf.where(not_evalated_mask), shape=(-1,)), dtype=tf.int32)

            if not_evalated_indices.shape[0] == 0:
                complete_op_results = op_results
            else:
                new_op_results = tf.stack([
                    self._ops[index_eval](x, training=training) for index_eval in not_evalated_indices
                ])

                complete_op_results = tf.concat([op_results, new_op_results], axis=0)

            if self._moving_average is None:
                element_shape = tf.shape(complete_op_results)[2:]
                self._moving_average = tf.zeros(shape=[mask_shape[0]] + element_shape)

            complete_op_results_indices = tf.concat([sampled_indices_on, not_evalated_indices], axis=0)
            self._moving_average, local_activation_estimate = tf.stop_gradient(
                moving_average_update_graph(self._lr, self._moving_average, complete_op_results,
                                            complete_op_results_indices)
            )

            dop_sum_reshaped = tf.reshape(tf.reduce_sum(dop_result, axis=0), shape=(1, -1))
            ma_reshaped = tf.reshape(local_activation_estimate, shape=(tf.shape(local_activation_estimate)[0], -1))
            dmask = tf.reduce_sum(tf.math.multiply(dop_sum_reshaped, ma_reshaped), axis=-1)

            dop_results = tf.repeat(tf.expand_dims(dop_result, axis=0), repeats=self._num_on_samples, axis=0)

            return None, None, dop_results, dmask, None, None

        op_result = tf.reduce_sum(op_results, axis=0)
        return op_result, backward


class BinaryProxylessNASStyleMixedOp(PartiallyEvaluatedMixOp, BinaryMixedOp):
    def __init__(self, operations: List[Callable], num_eval_samples: int, num_on_samples: int):
        super(BinaryProxylessNASStyleMixedOp, self).__init__(operations, num_eval_samples, num_on_samples)

        assert self._num_eval_samples > 1
        self.latest_mask_involved = None

    def stochastic_call(self, x, training, *args, **kwargs):
        return self.call_internal(
            x, training=training, num_eval_samples=self._num_eval_samples, num_on_samples=self._num_on_samples
        )

    def call_internal(self, x, training, num_eval_samples, num_on_samples):
        logits = self.clip_logits_gradient_by_active_indices(self._logits)

        weights = tf.math.softmax(logits, axis=-1)

        sampled_indices_eval, sampled_indices_on, mask = st_onehot_categorical(weights, num_eval_samples,
                                                                               num_on_samples)

        self.latest_mask_involved = tf.reduce_sum(tf.one_hot(sampled_indices_eval, depth=logits.shape[0]), axis=0)

        # this works because sampled_indices_on and sampled_indices_eval are always sorted in ascending order.
        op_results = tf.stack([self._ops[index_on](x, training=training) for index_on in sampled_indices_on])

        op_result = self.ops_times_mask_binary(x, training, op_results, mask, sampled_indices_on, sampled_indices_eval)
        return op_result

    @tf.custom_gradient
    def clip_logits_gradient_by_active_indices(self, x):
        def backward(dx):
            mask = self.latest_mask_involved

            new_logits = tf.boolean_mask(x - dx, mask)
            old_logits = x

            offset = tf.math.log(
                tf.reduce_sum(tf.math.exp(new_logits)) / tf.reduce_sum(tf.math.exp(old_logits))
            )

            dx += mask * offset

            return dx

        return x, backward
