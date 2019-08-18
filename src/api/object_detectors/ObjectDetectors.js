import ImageNetDetect from './ImageNetDetect';
import CocoSSdDetect from './CocoSddDetect';

export default class ObjectDetectors {

    constructor(image, type) {
        this.inputImage = image;
        this.type = type;
    }

    async process() {
        if(this.type == "mobilenet") {
            const imageNetDetect = new ImageNetDetect(this.inputImage);
            return await imageNetDetect.process();
        } else {
            const cocossd = new CocoSSdDetect(this.inputImage);
            return await cocossd.process();
        }
    }
}