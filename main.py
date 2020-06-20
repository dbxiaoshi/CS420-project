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
#仅保存性能最好的模型
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#设置训练参数，step200，epoch2
model.fit_generator(myGene,steps_per_epoch=200,epochs=2,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#保存结果
saveResult("data/membrane/test",results)