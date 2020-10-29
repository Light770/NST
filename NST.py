import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import image as img
from NST_utils import *
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pprint



CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS = ((img.imread(CONFIG.CONTENT_IMAGE)).shape)

# GRADED FUNCTION: compute_content_cost
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]), perm=[0,2,1])
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]), perm=[0,2,1])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) * (1/(4 * n_C * n_H * n_W))
    
    return J_content

# GRADED FUNCTION: gram_matrix
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, A, transpose_b = True)
    
    return GA

# GRADED FUNCTION: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S, perm = [3,2,1,0]), [n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm = [3,2,1,0]), [n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1/(4 * (n_H * n_W)**2 * n_C**2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

# GRADED FUNCTION: total_cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha * J_content + beta * J_style
    
    return J

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


"""!!! TENSORFLOW ISSUE"""
tf.reset_default_graph()
sess = tf.InteractiveSession()

# !!! shape : (1200, 960, 3)
content_image = img.imread(CONFIG.CONTENT_IMAGE)
# !!! shape : (1, 1200, 960, 3)
content_image = reshape_and_normalize_image(content_image)

# !!! shape : (948, 721, 3)
style_image = img.imread(CONFIG.STYLE_IMAGE)
# !!! shape : (1, 948, 721, 3)
style_image = reshape_and_normalize_image(style_image)


generated_image = generate_noise_image(content_image)

plt.imshow(generated_image[0])




model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

"""!!! TENSORFLOW ISSUE"""
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out

# No problems
J_content = compute_content_cost(a_C, a_G)

"""!!! TENSORFLOW ISSUE"""
sess.run(model['input'].assign(style_image))

# No problems 
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha = 2, beta = 11)



"""!!! TENSORFLOW ISSUE"""
optimizer = tf.train.AdamOptimizer(0.5)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 30000):
    
    """!!! TENSORFLOW ISSUE"""
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    

    
    for i in range(num_iterations):
    
        """!!! TENSORFLOW ISSUE"""
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        


        #COULD BE SOLVED AS TF2
        if i%200 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            """!!! NEED TO MAKE THE SAVE FUNCTION"""
            # save current generated image in the "/output" directory
            save_image("output/" + "myPhoto" + str(i) + ".png", generated_image)
    
    """!!! NEED TO MAKE THE SAVE FUNCTION"""
    # save last generated image
    save_image('output/generated_image2.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)