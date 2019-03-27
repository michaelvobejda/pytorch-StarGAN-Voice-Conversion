import os




def add_synth(dir_):
    for filename in os.listdir(dir_):

        if filename.startswith('SYNTH_'):
            continue

        src = dir_ + '/' + filename
        #dst ='xyz'+ dst
        dst = dir_ + '/' + 'SYNTH_' + filename

        os.rename(src, dst)





[add_synth(f'data/VCTK-Corpus/synth_audio/p{n}') for n in [226, 227, 228, 229, 247]]


