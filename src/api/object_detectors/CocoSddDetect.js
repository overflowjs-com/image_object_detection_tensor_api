// import tf from '@tensorflow/tfjs-node';
// import mobilenet from '@tensorflow-models/mobilenet';
import jpeg from 'jpeg-js';

const tf = require('@tensorflow/tfjs-node');
const cocossd = require('@tensorflow-models/coco-ssd');

// let cocomodal = null;

// cocossd.load({
//     base: 'mobilenet_v2'
// }).then((modal) => {
//     console.log("Modal is ready");
//     cocomodal = modal;
// })

export default class ImageNetDetect {

    constructor(image) {
        this.inputImage = image;
    }
    
    range(n) {
        return Array.from(Array(n).keys())
    }
    async loadModal() {
        const modal = await cocossd.load({
            base: 'mobilenet_v2'
        })
        return modal;
    }

    getInputImage() {

        const imageData = this.inputImage.replace('data:image/jpeg;base64','')
                            .replace('data:image/png;base64','');
        const buf = Buffer.from(imageData, 'base64');
        const pixels = jpeg.decode(buf, true);
        return pixels;
    }

    getImageByteArray(image, numOfChannels) {

        const pixels = image.data;
        const numOfPixels = image.width * image.height;
        const values = new Int32Array(numOfPixels*numOfChannels);

        this.range(0, numOfPixels).map((count) =>  {
            this.range(0, numOfChannels).map((channel) => {
                values[count*numOfChannels+channel] = pixels[count*4+channel];
            })
        })

        return values;
    }

    convertImageInput(image, channels) {
        
        const values = this.getImageByteArray(image, channels);
        const outShape = [image.height, image.width, channels];
        const input = tf.tensor3d(values, outShape, 'int32');
        
        return input;

    }


    async process() {
       
       const image = this.getInputImage();

       const model =  await this.loadModal();
     
       const inputValues = this.convertImageInput(image, 3);
       const predictions = await model.detect(inputValues);
       
       inputValues.dispose();
       return {data: predictions, type: "coco-ssd"};

    }
}