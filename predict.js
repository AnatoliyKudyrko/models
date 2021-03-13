    const tf = require("@tensorflow/tfjs"),
    tfnode = require("@tensorflow/tfjs-node"),
    fs = require("fs"),
    path = require("path");

async function main() {
    const model = await tf.loadLayersModel("file://./model/model.json");
    let file = "data/seg_test/seg_test/buildings/20057.jpg";
    let buffer = fs.readFileSync(file);
    let tfimage = tfnode.node.decodeImage(buffer, chanels=3);
    tfimage = tf.image.resizeBilinear(tfimage, [28, 28]);
    tfimage = tfimage.cast("float32").div(255);

    const pred = model.predict(tf.stack([tfimage]));
    pred.print();
}

main();

