//CONSTANTS


const fs = require("fs")
const getPixles = require("get-pixels")
const tf = require("@tensorflow/tfjs-node")
const { jStat } = require('jstat')



class Data {

    constructor(directory, size, parts){
        this.dir = directory
        this.size = size
        this.parts = parts
        this.startIndex = 0
        this.endIndex = size/parts
    }
    
    
    //PROBLEMS
    //-shuffle? check books!
    //tf.util.shuffle chekck!!!
    
    //IDEAS
    //Batch or PNG class?
    //

    //only shuffle batches to make sure all pictures are loaded

    //returns a Promise: when resolves returns a 3d array 
    //with arrays of pixel arrays and labels 
    //[[[pixel array 1], [lbl]], [[pixel array 2],[lbl]],...]
    //of all the pictures
    async loadData(){
        return new Promise((res, rej) => {
            let pngs = []
            let lbls = []
            fs.readdir(this.dir, async (err, data) => {
                if(err) return rej(err)
                else{
                    if(this.endIndex > this.size){
                        console.log("endIndex bigger than size of dataset")
                        return rej("EndIndex bigger than size of dataset")
                    }
                    else{
                        for(let i = this.startIndex; i < this.endIndex;i++){
                            pngs.push(await this.loadPNG(this.dir +  "/" + data[i]))
                            //split the name to get the "number.png", split again to get "number", convert to int
                            lbls.push(Number([data[i].split("_")[1].split(".")[0]]))
                        }

                        //special splitting part for train dataset
                        //because it cant be loaded fully
                        if(this.parts != 1){
                            if(this.endIndex = this.size){
                                this.startIndex = 0
                                this.endIndex = this.size/this.parts
                            }
                            else{
                                this.startIndex = this.endIndex
                                this.endIndex += this.size/this.parts
                            }
                        }

                        this.data = tf.tensor2d(this.normalizePictures(pngs))
                        this.labels = tf.tensor2d(this.prepLabels(lbls))
                        res();
                    }
                   
                }
            })
        })
    }

    clearData(){
        this.data = 0
        this.lables = 0
    }

    //returns an 2darray of shape [lbls.length, 47] [[0,0,...,0], [0,0,0,...,0], ...]
    prepLabels(lbls){
        let tensLbls = [];
        for(let lbl of lbls){
            let ts = []
            for(let i = 0; i<47; i++){
                ts.push(0)
            }
            ts[lbl] = 1
            tensLbls.push(ts)    
        }
        return tensLbls     
    }

    //normalize pictures from range 0-255 to 0-1
    normalizePictures(pics){
        let picsNorm = []
        //pics 0-255
        for(let i = 0; i < pics.length;i++){
            let pixels = []
            for (let j = 0; j < pics[i].length; j++) {
                 pixels.push(pics[i][j]/255)  
            }
            picsNorm.push(pixels)
        }
        return picsNorm
    }
    
    //transforms a original blackscale 4 channel image into and and 1d pixel grayscale array
    transform(pixArr){
        let rgbArr = new Array();
        for(let i = 0; i<pixArr.length/4;i++){
            rgbArr[i] = pixArr[i*4]
        }
        return rgbArr
    }

    //returns a Promise : when resolved 1d pixel array is returned, rejected -> error
    async loadPNG(path){
        return new Promise((res, rej)=>{
            let dat = this // this not usable in getPixels function
            getPixles(path, function(err, pixels) {
                if(err) return rej(err);
                res(dat.transform(pixels.data));
            });
        })
        
    }
}

module.exports = {
    Data
}