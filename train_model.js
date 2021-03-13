
const tf = require("@tensorflow/tfjs"),
      tfnode = require("@tensorflow/tfjs-node"),
      fs = require("fs"),
      path = require("path");


function loadDataset(baseDir="./data/seg_train/seg_train") {
    const folders = fs.readdirSync(baseDir);
    const allFiles = [];
    for (const i in folders) {
        const folder = folders[i];
        const files = fs.readdirSync(path.join(baseDir, folder));
        allFiles.push(...files.map(f => {
            return [
                path.join(baseDir, folder, f),
                i
            ]
        }));
    }

    let images = [];
    let labels = [];
    for (const [file, i] of allFiles) {
        let buffer = fs.readFileSync(file);
        let tfimage = tfnode.node.decodeImage(buffer, chanels=3);
        tfimage = tf.image.resizeBilinear(tfimage, [28, 28]),
        tfimage = tfimage.cast("float32").div(255);
        images.push(tfimage);
        labels.push(i);
    }
    return [
        tf.stack(images),
        tf.oneHot(tf.tensor1d(labels, 'int32'), 6)
    ]
}

async function main() {
    const [xs, ys] = loadDataset();

    const model = tf.sequential({
        layers: [
            tf.layers.conv2d({
                inputShape: [28, 28, 3], 
                filters: 48,
                kernelSize: 3,
                activation: "relu"
            }),
            tf.layers.dense({
                inputShape: [1],
                units: 1,
                activation: "relu",
                kernelInitializer: "ones"
            }),
            tf.layers.dense({
                inputShape: [1],
                units: 1,
                activation: "relu",
                kernelInitializer: "zeros"
            }),
            tf.layers.maxPool2d({
                poolSize: [2, 2],
                strides: [1, 1]
            }),
            tf.layers.flatten(),
            tf.layers.dropout(0.1),
            tf.layers.dense({
                units: 6,
                activation: "softmax"
            })
    
        ]
       });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ["accuracy"]
    });

    const h = await model.fit(xs, ys, {
       epochs: 100,
       batchSize: 150
    });
    await model.save('file://./model');
}

main();