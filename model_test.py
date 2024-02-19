
import time
from ops import *
from ssim import *



class T_CNN(object):

    def __init__(self,
                 sess,
                 image_height=128,
                 image_width=128,
                 label_height=128,
                 label_width=128,
                 batch_size=1,
                 c_dim=3,
                 c_depth_dim=1,
                 c_ctran_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None,
                 test_image_name=None,
                 test_depth_name=None,
                 test_ctran_name=None,
                 id=None
                 ):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.dropout_keep_prob = 0.5
        self.test_image_name = test_image_name
        self.test_depth_name = test_depth_name
        self.test_ctran_name = test_ctran_name
        self.id = id
        self.c_dim = c_dim
        self.df_dim = 64
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.c_depth_dim = c_depth_dim
        self.c_ctran_dim = c_ctran_dim
        self.new_height = 0
        self.new_width = 0
        self.new_height_half = 0
        self.new_width_half = 0
        self.new_height_half_half = 0
        self.new_width_half_half = 0
        image_test = get_image(self.test_image_name, is_grayscale=False)
        shape = image_test.shape
        RGB = Image.fromarray(np.uint8(image_test * 255))
        RGB1 = RGB.resize(((shape[1] // 8 - 0) * 8, (shape[0] // 8 - 0) * 8))
        image_test = np.asarray(np.float32(RGB1) / 255)
        shape = image_test.shape
        self.new_height = shape[0]
        self.new_width = shape[1]

        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                     name='images')
        self._depth = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_height, self.image_width, self.c_depth_dim],
                                     name='depth')
        self.ctran = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_ctran_dim],
                                     name='ctran')
        self.pred_h = self.model()
        self.saver = tf.train.Saver()

    def train(self, config):

        # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
        image_test = get_image(self.test_image_name, is_grayscale=False)
        shape = image_test.shape
        RGB = Image.fromarray(np.uint8(image_test * 255))
        RGB1 = RGB.resize(((shape[1] // 8 - 0) * 8, (shape[0] // 8 - 0) * 8))
        image_test = np.asarray(np.float32(RGB1) / 255)
        shape = image_test.shape

        expand_test = image_test[np.newaxis, :, :, :]
        expand_zero = np.zeros([self.batch_size - 1, shape[0], shape[1], shape[2]])
        batch_test_image = np.append(expand_test, expand_zero, axis=0)

        depth_test = get_image(self.test_depth_name, is_grayscale=False)
        shape = depth_test.shape
        Depth = Image.fromarray(np.uint8(depth_test * 255))
        Depth1 = Depth.resize(((shape[1] // 8 - 0) * 8, (shape[0] // 8 - 0) * 8))
        depth_test = np.asarray(np.float32(Depth1) / 255)
        shape1 = depth_test.shape

        expand_test1 = depth_test[np.newaxis, :, :]
        expand_zero1 = np.zeros([self.batch_size - 1, shape1[0], shape1[1]])
        batch_test_depth1 = np.append(expand_test1, expand_zero1, axis=0)
        batch_test_depth = batch_test_depth1.reshape(self.batch_size, shape1[0], shape1[1], 1)
        tf.global_variables_initializer().run()

        ctran_test = get_image(self.test_ctran_name, is_grayscale=False)
        shape = ctran_test.shape
        Ctran = Image.fromarray(np.uint8(ctran_test * 255))
        Ctran1 = Ctran.resize(((shape[1] // 8 - 0) * 8, (shape[0] // 8 - 0) * 8))
        ctran_test = np.asarray(np.float32(Ctran1) / 255)
        shape2 = ctran_test.shape

        expand_test2 = ctran_test[np.newaxis, :, :]
        expand_zero2 = np.zeros([self.batch_size - 1, shape2[0], shape2[1]])
        batch_test_ctran1 = np.append(expand_test2, expand_zero2, axis=0)
        batch_test_ctran = batch_test_ctran1.reshape(self.batch_size, shape2[0], shape2[1], 1)
        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        start_time = time.time()
        result_h = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image, self._depth: batch_test_depth, self.ctran: batch_test_ctran})
        all_time = time.time()
        final_time = all_time - start_time
        print(final_time)

        _, h, w, c = result_h.shape
        for id in range(0, 1):
            result_h0 = result_h[id, :, :, :].reshape(h, w, 3)
            image_path0 = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path0, self.test_image_name) #+ '_out.png'
            imsave_lable(result_h0, image_path)

    def model(self):

        with tf.variable_scope("fusion_branch") as scope1:
            depth_2down = max_pool_2x2(1 - self._depth)
            depth_4down = max_pool_2x2(depth_2down)
            depth_8down = max_pool_2x2(depth_4down)

            # first RGB encoder
            conv2_RDCP = conv2d(1 - self._depth, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_RDCP")
            conv2_1 = tf.nn.relu(conv2d(self.images, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1"))
            conv2_cb1_1 = tf.nn.relu(
                conv2d(tf.add(conv2_1, tf.multiply(conv2_1, conv2_RDCP)), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,
                       name="conv2_cb1_1"))
            conv2_cb1_2 = tf.nn.relu(
                conv2d(conv2_cb1_1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2"))
            conv2_cb1_3 = conv2d(conv2_cb1_2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3")
            first_add = tf.add(conv2_1, conv2_cb1_3)
            conv2_2 = tf.nn.relu(
                conv2d(first_add, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2"))
            conv2_cb1_4 = tf.nn.relu(
                conv2d(conv2_2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4"))
            conv2_cb1_5 = tf.nn.relu(
                conv2d(conv2_cb1_4, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5"))
            conv2_cb1_6 = conv2d(conv2_cb1_5, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6")
            first_add1 = tf.add(conv2_2, conv2_cb1_6)
            encoder1_down2 = max_pool_2x2(first_add1)

            # second RGB encoder
            conv2_RDCP_2 = max_pool_2x2(
                conv2d(1 - self._depth, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_RDCP_2"))
            conv2_2_1 = tf.nn.relu(conv2d(encoder1_down2, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1"))
            conv2_cb2_1 = tf.nn.relu(
                conv2d(tf.add(conv2_2_1, tf.multiply(conv2_2_1, conv2_RDCP_2)), 3, 256, k_h=3, k_w=3, d_h=1,
                       d_w=1, name="conv2_cb2_1"))
            conv2_cb2_2 = tf.nn.relu(
                conv2d(conv2_cb2_1, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2"))
            conv2_cb2_3 = conv2d(conv2_cb2_2, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3")
            second_add = tf.add(conv2_2_1, conv2_cb2_3)
            conv2_2_2 = tf.nn.relu(
                conv2d(second_add, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2"))
            conv2_cb2_4 = tf.nn.relu(
                conv2d(conv2_2_2, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4"))
            conv2_cb2_5 = tf.nn.relu(
                conv2d(conv2_cb2_4, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5"))
            conv2_cb2_6 = conv2d(conv2_cb2_5, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6")
            second_add1 = tf.add(conv2_2_2, conv2_cb2_6)
            encoder2_down2 = max_pool_2x2(second_add1)

            # third RGB encoder
            conv2_RDCP_3 = max_pool_2x2(max_pool_2x2(
                conv2d(1 - self._depth, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_RDCP_3")))
            conv2_3_1 = tf.nn.relu(conv2d(encoder2_down2, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1"))
            conv2_cb3_1 = tf.nn.relu(
                conv2d(tf.add(conv2_3_1, tf.multiply(conv2_3_1, conv2_RDCP_3)), 3, 512, k_h=3, k_w=3, d_h=1,
                       d_w=1, name="conv2_cb3_1"))
            conv2_cb3_2 = tf.nn.relu(
                conv2d(conv2_cb3_1, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2"))
            conv2_cb3_3 = conv2d(conv2_cb3_2, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3")
            third_add = tf.add(conv2_3_1, conv2_cb3_3)
            conv2_3_2 = tf.nn.relu(
                conv2d(third_add, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2"))
            conv2_cb3_4 = tf.nn.relu(
                conv2d(conv2_3_2, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4"))
            conv2_cb3_5 = tf.nn.relu(
                conv2d(conv2_cb3_4, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5"))
            conv2_cb3_6 = conv2d(conv2_cb3_5, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6")
            third_add1 = tf.add(conv2_3_2, conv2_cb3_6)

            ########### concate

            channle_weight_third_con_temp = self.Squeeze_excitation_layer(third_add1, out_dim=512, ratio=16,
                                                                          layer_name='channle_weight_third_con_temp')
            third_con_ff = tf.nn.relu(
                conv2d(channle_weight_third_con_temp, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="third_con_ff"))

            channle_weight_second_con_temp = self.Squeeze_excitation_layer(second_add1, out_dim=256, ratio=16,
                                                                           layer_name='channle_weight_second_con_temp')
            second_con_ff = tf.nn.relu(
                conv2d(channle_weight_second_con_temp, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="second_con_ff"))

            channle_weight_first_con_temp = self.Squeeze_excitation_layer(first_add1, out_dim=128, ratio=16,
                                                                          layer_name='channle_weight_first_con_temp')
            first_con_ff = tf.nn.relu(
                conv2d(channle_weight_first_con_temp, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="first_con_ff"))
            #############################################################################################################
            ctran_2down = max_pool_2x2(1 - self.ctran)
            ctran_4down = max_pool_2x2(ctran_2down)
            ctran_8down = max_pool_2x2(ctran_4down)

            # first decoder
            decoder_input = tf.add(third_con_ff, tf.multiply(third_con_ff, ctran_4down))
            conv2_1_1_dc = tf.nn.relu(conv2d(decoder_input, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_1_dc"))
            conv2_dc1_1 = tf.nn.relu(conv2d(conv2_1_1_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_1"))
            conv2_dc1_2 = tf.nn.relu(conv2d(conv2_dc1_1, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_2"))
            conv2_dc1_3 = conv2d(conv2_dc1_2, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_3")
            first_dadd = tf.add(conv2_1_1_dc, conv2_dc1_3)
            conv2_1_2_dc = tf.nn.relu(conv2d(first_dadd, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_2_dc"))
            conv2_dc1_4 = tf.nn.relu(conv2d(conv2_1_2_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_4"))
            conv2_dc1_5 = tf.nn.relu(conv2d(conv2_dc1_4, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_5"))
            conv2_dc1_6 = conv2d(conv2_dc1_5, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_6")
            first_dadd1 = tf.add(conv2_1_2_dc, conv2_dc1_6)
            decoder1_down2 = tf.image.resize_bilinear(first_dadd1, [second_con_ff.get_shape().as_list()[1],
                                                                    second_con_ff.get_shape().as_list()[2]])

            # second decoder
            decoder_input1 = tf.add(second_con_ff, tf.multiply(second_con_ff, ctran_2down))
            concate_1 = tf.concat(axis=3, values=[decoder1_down2, decoder_input1])
            conv2_1_3_dc = tf.nn.relu(conv2d(concate_1, 512, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_3_dc"))
            conv2_dc2_1 = tf.nn.relu(conv2d(conv2_1_3_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_1"))
            conv2_dc2_2 = tf.nn.relu(conv2d(conv2_dc2_1, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_2"))
            conv2_dc2_3 = conv2d(conv2_dc2_2, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_3")
            second_dadd = tf.add(conv2_1_3_dc, conv2_dc2_3)
            conv2_1_4_dc = tf.nn.relu(conv2d(second_dadd, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_4_dc"))
            conv2_dc2_4 = tf.nn.relu(conv2d(conv2_1_4_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_4"))
            conv2_dc2_5 = tf.nn.relu(conv2d(conv2_dc2_4, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_5"))
            conv2_dc2_6 = conv2d(conv2_dc2_5, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_6")
            second_dadd1 = tf.add(conv2_1_4_dc, conv2_dc2_6)
            decoder2_down2 = tf.image.resize_bilinear(second_dadd1, [first_con_ff.get_shape().as_list()[1],
                                                                     first_con_ff.get_shape().as_list()[2]])
            # third decoder
            decoder_input2 = tf.add(first_con_ff, tf.multiply(first_con_ff, (1 - self.ctran)))
            concate_2 = tf.concat(axis=3, values=[decoder2_down2, decoder_input2])
            conv2_1_5_dc = tf.nn.relu(conv2d(concate_2, 256, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_5_dc"))
            conv2_dc3_1 = tf.nn.relu(conv2d(conv2_1_5_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_1"))
            conv2_dc3_2 = tf.nn.relu(conv2d(conv2_dc3_1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_2"))
            conv2_dc3_3 = conv2d(conv2_dc3_2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_3")
            third_dadd = tf.add(conv2_1_5_dc, conv2_dc3_3)
            conv2_1_6_dc = tf.nn.relu(conv2d(third_dadd, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_6_dc"))
            conv2_dc3_4 = tf.nn.relu(conv2d(conv2_1_6_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_4"))
            conv2_dc3_5 = tf.nn.relu(conv2d(conv2_dc3_4, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_5"))
            conv2_dc3_6 = conv2d(conv2_dc3_5, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_6")
            third_dadd1 = tf.add(conv2_1_6_dc, conv2_dc3_6)
            conv2_refine = conv2d(third_dadd1, 128, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_refine")

            final_results = conv2_refine

        return final_results

    def save(self, checkpoint_dir, step):
        model_name = "coarse.model"
        model_dir = "%s_%s" % ("coarse", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("coarse", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            scale = input_x * excitation

            return scale


def unsqueeze2d(x, factor=2):
    x = tf.depth_to_space(x, factor)
    return x
