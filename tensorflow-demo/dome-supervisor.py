import tensorflow as tf

"""
    1）自动去checkpoint加载数据或初始化数据 ，因此我们就不需要手动初始化或者从checkpoint中加载数据
    2）自身有一个Saver，可以用来保存checkpoint，因此不需要创建Saver，直接使用Supervisor里的Saver即可
    3）有一个summary_computed用来保存Summary，因此不需要创建summary_writer
"""
a = tf.Variable(1)
b = tf.Variable(2)
c = tf.add(a, b)
update = tf.assign(a, c)


tf.summary.scalar("a", a)
init_op = tf.initialize_all_variables()
merged_summary_op = tf.summary.merge_all()

sv = tf.train.Supervisor(logdir="./tmp/", init_op=init_op)  # 直接初始化
saver = sv.saver
with sv.managed_session() as sess:
    for i in range(10000):
        update_ = sess.run(update)
        if i % 10 == 0:
            merged_summary = sess.run(merged_summary_op)
            sv.summary_computed(sess, merged_summary, global_step=i)  # 直接将summary保存

        if i % 100 == 0:
            saver.save(sess, save_path="./tmp/", global_step=i)  # 直接将模型参数保存





