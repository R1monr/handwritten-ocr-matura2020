//LIBRARIES//
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs")

const {Data} = require("./data.js")

//CONSTANTS//
const  DIR_TRAIN= "C:/Users/Ramon/Documents/school/Maturaarbeit/Pictures/balanced/train";
const  DIR_VALIDATE= "C:/Users/Ramon/Documents/school/Maturaarbeit/Pictures/balanced/validate";
const  DIR_TEST= "C:/Users/Ramon/Documents/school/Maturaarbeit/Pictures/balanced/test";

const DIR_LOG = "./logs"
const DIR_STATS = "./stats"

const STRUCTURE = "100-100-200-100-75   "

const NAME = "31.12.20-" + STRUCTURE
const STATS_FILE = DIR_STATS +"/" + NAME + ".txt"

const TRAIN_SIZE = 94000 //47000 possible to load with max-old-space-size=4000
const VALIDATE_SIZE = 18800 //18800
const TEST_SIZE = 18800

const ACCURACY_THRESHOLD = 0.6

const BATCH_SIZE = 50;
const DROPOUT_RATE = 0.3
const EPOCHS =  1
const EPOCHS_PER_FIT = 2


let history //= [[],[],[],[]];
let pred;
let experiment;
let timeToPreviousExperiment = 0;

//const summaryWriter = tf.node.summaryFileWriter(DIR_LOG)

//process.memoryUsage()

// Tnesor2d = tf.math.confusionMatrix(labels,predictions,classes)

//Future Data objects declared here for global scope
let train;
let validate;
let test;


async function superrun(){
    for(experiment = 1; experiment < 2; experiment++){
        await run()
    }
}

async function run(){
    train = new Data(DIR_TRAIN, TRAIN_SIZE,2)
    validate = new Data(DIR_VALIDATE, VALIDATE_SIZE,2)
    test = new Data(DIR_TEST, TEST_SIZE,1)

    const model = createModel()

    for(let i = 0; i<EPOCHS; i++){
        await train.loadData()
        await validate.loadData()
        console.log("NEW FIT STARTED")
        console.log("epoch part: ", i+1, " out of ", EPOCHS)
        model.fit(train.data, train.labels, {
            epochs: EPOCHS_PER_FIT,
            batchSize: BATCH_SIZE,
            callbacks: [tf.node.tensorBoard(DIR_LOG)], //, tf.callbacks.earlyStopping({monitor: "val_acc", patience: 8, verbose: true})
            validationData: [validate.data,validate.labels],
            shuffle: true
        })
        .then(h => {
            // history[0].push(h.history.val_loss[0])
            // history[0].push(h.history.val_loss[1])
            // history[1].push(h.history.val_acc[0])
            // history[1].push(h.history.val_acc[1])
            // history[2].push(h.history.loss[0])
            // history[2].push(h.history.loss[1])
            // history[3].push(h.history.acc[0])
            // history[3].push(h.history.acc[1])
            history = h.history
        })
        .catch(err =>{
            console.log("ERROR", err)
        })
    }
    //clear Memory of train Dataset
    train.clearData()
    validate.clearData()

    await test.loadData()

    const prediction = model.predict(test.data, {batchSize: BATCH_SIZE, verbose: true})
    pred = accuracyTest(prediction) 
    console.log(pred, " accuracy over TEST_SIZE " + TEST_SIZE)
    saveStats(STATS_FILE)

    const saveResults = await model.save("file://model-test")
    
 }
 
 async function run2(){
 }

 //{onBatchBegin: onBBegin(), onBatchEnd}

function createModel(){
    const model = tf.sequential()  

    //Layer 1
    model.add(tf.layers.dense({
        units: 100,
        activation: "elu",
        useBias:true,
        inputShape: [784],
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

    //Layer 2
    model.add(tf.layers.dense({
        activation: "elu",
        units: 100,
        useBias:true,
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))
    
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

    //Layer 3
    model.add(tf.layers.dense({
        activation: "elu",
        units: 200,
        useBias:true,
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

    //Layer 4
    model.add(tf.layers.dense({
        activation: "elu",
        units: 1,
        useBias:true,
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

     //Layer 5
     model.add(tf.layers.dense({
        activation: "elu",
        units: 75,
        useBias:true,
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

    //Layer 6
    model.add(tf.layers.dense({
        activation: "softmax",
        units: 47,
        useBias:true,
        kernelInitializer: "heNormal",
        biasInitializer: "heNormal"

    }))

    model.compile({
        optimizer: "adam", //tf.train.adam(LEARNING_RATE),
        loss: 'categoricalCrossentropy',
        metrics: ["accuracy",]
    })

    model.summary()
    
    return model

}

function accuracyTest(prediction){
    let tru = 0
    let predArr = prediction.arraySync()
    let testArr = test.labels.arraySync()
    for(let i = 0; i < testArr.length; i++){
        for(let j = 0; j < testArr[i].length;j++){
            if (predArr[i][j]> ACCURACY_THRESHOLD && testArr[i][j] == 1){
                tru = tru + 1
                break
            }
        }
    }
    return tru/TEST_SIZE
}

async function saveStats(file){
    let stats = [
        "TRY " + experiment.toString() + "\n",
        "Parameters\n",
        "Structure: " + STRUCTURE + "\n",
        "Epochs: " + EPOCHS.toString() + "\n",
        "Epochs_per_Fit: " + EPOCHS_PER_FIT.toString() + "\n",
        "Number_of_Parameters: " + calcParameters().toString() + "\n",
        "time_of_Try: " + calcTime(false) + " " + (process.uptime() - timeToPreviousExperiment).toString() + "\n",
        "total_time: " + calcTime(true) + " " + process.uptime().toString() + "\n",
        "Batch size: " + BATCH_SIZE.toString() + "\n",
        "Accuracy treshold: " + ACCURACY_THRESHOLD.toString() + "\n",
        "Dropout Rate: " + DROPOUT_RATE.toString() + "\n\n",
        "Results\n",
        "train_loss: " + history.loss[history.loss.length-1].toString() + "\n",
        "train_acc: " + history.acc[history.acc.length-1].toString() + "\n",
        "val_loss: " + history.val_loss[history.val_loss.length-1].toString() + "\n",
        "val_acc: " + history.val_acc[history.val_acc.length-1].toString() + "\n",
        "test_acc: " + pred.toString() + "\n",
        "test_difference: " + (history.acc[history.acc.length-1] - pred).toString() + "\n",
        "Progress_over_last_5_epochs: " + (history.acc[history.acc.length-1] - history.acc[history.acc.length-6]).toString() + "\n\n\n\n\n"
    ]

        
    for(let i of stats){
        fs.appendFileSync(file, i)
    }
    console.log("saving statistics successful")
    
    
}

function calcParameters(){
    let layers = [784]
    let parameters = 0
    for(let layer of STRUCTURE.split("-")){
        layers.push(Number(layer))
    }
    for(let i = 1; i<layers.length;i++){
        parameters += layers[i-1]*layers[i]
    }
    parameters += layers[layers.length-1]* 47
    return parameters
}

function calcTime(total){
    let time = []
    let seconds
    if (total){
        seconds = process.uptime()
        timeToPreviousExperiment = seconds
    }
    else{
        seconds = process.uptime() - timeToPreviousExperiment
    }
    
    let timeStr = ""
    time.push((Math.floor(seconds / 3600)).toString())
    seconds -= time[0]*3600
    time.push((Math.floor(seconds / 60)).toString())
    seconds -= time[1]*60
    time.push(seconds.toString())

    for(let part of time){
        timeStr += part.toString() + " "
    }
    return timeStr
}


superrun()

//run()

//run2()


