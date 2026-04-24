from PIL import Image
import os
p='evaluation/plots/explain/propagation_node0_0_4.gif'
if not os.path.exists(p):
    print('MISSING_GIF')
else:
    im=Image.open(p)
    i=0
    outdir='evaluation/plots/explain/prop_frames'
    os.makedirs(outdir, exist_ok=True)
    try:
        while True:
            frame = im.convert('RGBA')
            fname=os.path.join(outdir,f'propagation_node0_0_4-{i:03d}.png')
            frame.save(fname)
            print('SAVED',fname)
            i+=1
            im.seek(im.tell()+1)
    except EOFError:
        print('DONE',i,'frames')
