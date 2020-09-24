from bert_serving.client import BertClient
music_sentences = ['Music is an art form, and cultural activity, whose medium is sound', 'In its most general form, the activities describing music as an art form or cultural activity include the creation of works of music (songs, tunes, symphonies, and so on), the criticism of music, the study of the history of music, and the aesthetic examination of music.', 'There are many types of music, including popular music, traditional music, art music, music written for religious ceremonies and work songs such as chanteys', 'Music can be divided into genres (e.g., country music) and genres can be further divided into subgenres (e.g., country blues and pop country are two of the many country subgenres), although the dividing lines and relationships between music genres are often subtle, sometimes open to personal interpretation, and occasionally controversial.']
computer_sentences =  ['A computer is a machine that can be instructed to carry out sequences of arithmetic or logical operations automatically via computer programming.', 'The first digital electronic calculating machines were developed during World War II. The first semiconductor transistors in the late 1940s were followed by the silicon-based MOSFET (MOS transistor) and monolithic integrated circuit (IC) chip technologies in the late 1950s, leading to the microprocessor and the microcomputer revolution in the 1970s.' 'Peripheral devices include input devices (keyboards, mice, joystick, etc.), output devices (monitor screens, printers, etc.), and input/output devices that perform both functions (e.g., the 2000s-era touchscreen).']

common = [
    '-model_dir', '../uncased_L-12_H-768_A-12/',
    '-num_worker', '2',
    '-port', str(port),
    '-port_out', str(port_out),
    '-max_seq_len', '20',
    # '-client_batch_size', '2048',
    '-max_batch_size', '256',
    # '-num_client', '1',
    '-pooling_strategy', 'REDUCE_MEAN',
    '-pooling_layer', '-2',
    '-gpu_memory_fraction', '0.2',
    '-device','3',
]
args = get_args_parser().parse_args(common)

subset_vec_all_layers = []
for pool_layer in range(1, 13):
    setattr(args, 'pooling_layer', [-pool_layer])
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')
    bc = BertClient(port=port, port_out=port_out, show_server_config=True)
    subset_vec_all_layers.append(bc.encode(subset_text))
    bc.close()
    server.close()
    print('done at layer -%d' % pool_layer)


from sklearn.decomposition import PCA
pca_embed = [PCA(n_components=2).fit_transform(v) for v in subset_vec_all_layers]