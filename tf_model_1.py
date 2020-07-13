import data_organise
import tf_functions
import tensorflow as tf
import matplotlib.pyplot as plt


def tf_model_1(X_train, Y_train, X_test, Y_test, alpha=0.0001, num_epochs=300, minibatch_size=32, print_cost=True):
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    seed = 3

    X, Y = tf_functions.create_placeholders(n_x, n_y)
    parameters = tf_functions.initialize_parameters(n_x, n_y)
    Z3 = tf_functions.forward_propagation(X, parameters)
    cost = tf_functions.compute_cost_softmax(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            mini_batches = data_organise.random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: mini_batch_X, Y: mini_batch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost and epoch % 100 == 0:
                print('Cost after epoch {0} = {1}'.format(epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
        print('Parameters trained')

        # this line gives back the index of the largest value of Z3 and also the index of the largest value of Y and compares them for similarity.  If they are the same then correct_prediction gets an boolean 1, if not it gets a boolean 0
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        saver = tf.train.Saver()
        save_path = saver.save(sess, './parameters.ckpt')
        print("Model saved in path: %s" % save_path)

        return parameters
