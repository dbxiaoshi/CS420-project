from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#由于训练集较小，数据增强
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#生成增强后的数据
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
#使用CLR学习率
clr = CyclicLR(base_lr=0.0001, max_lr=0.0006,
                   step_size=2000., mode='exp_range',
                   gamma=0.99994)
model.fit_generator(myGene, batch_size=20, nb_epoch=1,callbacks=[clr])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#保存结果
saveResult("data/membrane/test",results)