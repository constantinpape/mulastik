<docs lang="markdown">

</docs>
<config lang="json">
{
    "name":"ImageAnnotator",
    "type":"window",
    "tags":[],
    "ui":"",
    "version":"0.5.64",
    "api_version":"0.1.5",
    "description":"An image annotator made with OpenLayers",
    "icon":"extension",
    "inputs":null,
    "outputs":null,
    "env":"",
    "requirements":[
        "https://cdn.jsdelivr.net/npm/vue@2.6.10/dist/vue.min.js",
        "https://cdn.jsdelivr.net/npm/openlayers@4.6.5/css/ol.css",
        "https://cdn.jsdelivr.net/npm/openlayers@4.6.5/dist/ol.min.js",
        "https://static.imjoy.io/spectre.css/spectre.min.css",
        "https://static.imjoy.io/spectre.css/spectre-exp.min.css",
        "https://static.imjoy.io/spectre.css/spectre-icons.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/13.1.5/nouislider.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/13.1.5/nouislider.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/pako/1.0.10/pako.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/upng-js/2.1.0/UPNG.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.2.0/jszip.min.js",
        "https://use.fontawesome.com/releases/v5.8.2/css/all.css"
    ],
    "dependencies":[
        "oeway/ImJoy-Plugins:ImageSelection",
        "oeway/ImJoy-Plugins:Tif File Importer"
    ],
    "defaults":{
        "w": 35,
        "h": 25,
        "fullscreen": true
    },
    "cover": "https://dl.dropbox.com/s/hmoio1d62vuvito/annotator-v0.5.40.gif"
}
</config>
<script lang="javascript">
const Map = ol.Map;
const View = ol.View;
const Draw = ol.interaction.Draw;
//const Modify = ol.interaction.Modify;
const Select = ol.interaction.Select;
const defaultInteractions = ol.interaction.defaults;
const TileLayer = ol.layer.Tile;
const Style = ol.style.Style;
const Fill = ol.style.Fill;
const Stroke = ol.style.Stroke;
const Text = ol.style.Text;
const VectorLayer = ol.layer.Vector;
const OSM = ol.source.OSM;
const VectorSource = ol.source.Vector;
const MousePosition = ol.control.MousePosition;
const LayerSwitcher = ol.control.LayerSwitcher;
const Zoomify = ol.source.Zoomify;
const Static = ol.source.ImageStatic;
const ImageLayer = ol.layer.Image;
const Projection = ol.proj.Projection;
const getCenter = ol.extent.getCenter;
const createStringXY = ol.coordinate.createStringXY;
const DragAndDrop = ol.interaction.DragAndDrop;
const GeoJSON = ol.format.GeoJSON;
const RasterSource = ol.source.Raster;
function pathJoin(...parts) {
    let separator = '/';
    let replace = new RegExp(separator + '{1,}', 'g');
    return parts.join(separator).replace(replace, separator);
}
function file2base64(file) {
    return new Promise((resolve, reject) => {
        var reader = new FileReader();
        reader.onload = (event) => {
            resolve(event.target.result)
        }
        reader.onerror = (err) => {
            reject(err)
        }
        reader.readAsDataURL(file);
    })
}
function file2arraybuffer(file) {
    return new Promise((resolve, reject) => {
        var reader = new FileReader();
        reader.onload = (event) => {
            resolve(event.target.result)
        }
        reader.onerror = (err) => {
            reject(err)
        }
        reader.readAsArrayBuffer(file);
    })
}
function randId(){
    return '_' + Math.random().toString(36).substr(2, 9)
}
function file2text(file) {
    return new Promise((resolve, reject) => {
        var reader = new FileReader();
        reader.onload = (event) => {
            resolve(event.target.result)
        }
        reader.onerror = (err) => {
            reject(err)
        }
        reader.readAsText(file, 'utf8');
    })
}
// Generate url for image
function array2url(imageArr, w, h, low, high, lut) {
    var canvas = document.createElement('canvas');
    canvas.width = w
    canvas.height = h
    var ctx = canvas.getContext("2d");
    var canvas_img = ctx.getImageData(0, 0, canvas.width, canvas.height)
    var canvas_img_data = canvas_img.data;
    var count = w * h
    var range = high - low
    var min = Number.POSITIVE_INFINITY,
        max = Number.NEGATIVE_INFINITY;
    var v;
    if (range <= 0) {
        for (let i = 0; i < count; i++) {
            if (imageArr[i] > max) max = imageArr[i]
            if (imageArr[i] < min) min = imageArr[i]
        }
        low = min
        high = max
        range = max - min
    }
    for (let i = 0; i < count; i++) {
        if (imageArr[i] > max) max = imageArr[i]
        if (imageArr[i] < min) min = imageArr[i]
        v = (imageArr[i] - low) / range * 255
        v = v > 255 ? 255 : v
        if (lut) {
            v = lut[parseInt(v)]
            canvas_img_data[i * 4] = v[0]
            canvas_img_data[i * 4 + 1] = v[1]
            canvas_img_data[i * 4 + 2] = v[2]
            canvas_img_data[i * 4 + 3] = 255
        } else {
            canvas_img_data[i * 4] = v
            canvas_img_data[i * 4 + 1] = v
            canvas_img_data[i * 4 + 2] = v
            canvas_img_data[i * 4 + 3] = 255
        }
    }
    ctx.putImageData(canvas_img, 0, 0);
    return [canvas.toDataURL("image/png"), min, max, low, high]
}
function getFileExtension(filename) {
    return (/[.]/.exec(filename)) ? /[^.]+$/.exec(filename)[0] : undefined;
}
async function getMeta(url) {
    return new Promise((resolve, reject) => {
        let img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}
async function url2base64(url) {
    return new Promise((resolve, reject) => {
        var img = new Image();
        img.crossOrigin = 'anonymous'
        img.onload = function () {
            var canvas = document.createElement('canvas')
            var ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            resolve({
                url: canvas.toDataURL("image/png"),
                w: img.width,
                h: img.height
            })
        }
        img.onerror = function (e) {
            reject('image load error')
        };
        img.src = url;
    });
}
function mkdir(dir) {
    return new Promise(async (resolve, reject) => {
        let paths = dir.split('/');
        let index = 1;
        async function next(index) {
            if (index > paths.length) return resolve();
            let newPath = paths.slice(0, index).join('/');
            if (newPath && !await exists(newPath)) {
                api.fs.mkdir(newPath, function (err) {
                    if (err) {
                        console.log('err: ', err)
                    }
                    next(index + 1)
                })
            } else {
                await next(index + 1);
            }
        }
        await next(index);
    })
}
function stat(dir) {
    return new Promise((resolve, reject) => {
        api.fs.stat(dir, (err, stats) => {
            if (err) {
                console.log('fs stat: ', err)
                reject(err)
            } else {
                resolve(stats)
            }
        })
    })
}
function exists(path) {
    return new Promise((resolve) => {
        api.fs.exists(path, (ext) => {
            resolve(ext)
        })
    })
}
function writeFile(path, content) {
    return new Promise((resolve, reject) => {
        api.fs.writeFile(path, content, (err, data) => {
            if (err) {
                reject(err)
            } else {
                resolve(data)
            }
        })
    })
}
function readFile(path, encoding) {
    return new Promise((resolve, reject) => {
        encoding = encoding === undefined ? 'utf8': encoding
        encoding = encoding === null ? undefined: encoding
        api.fs.readFile(path, encoding, (err, data) => {
            if (err) {
                reject(err)
            } else {
                resolve(data)
            }
        })
    })
}
function readdir(path) {
    return new Promise((resolve, reject) => {
        api.fs.readdir(path, (err, data) => {
            if (err) {
                reject(err)
            } else {
                resolve(data)
            }
        })
    })
}
function unlink(path) {
    return new Promise((resolve, reject) => {
        api.fs.unlink(path, (err) => {
            if (err) {
                reject(err)
            } else {
                resolve()
            }
        })
    })
}
const assert = function(condition, message) {
    if (!condition)
        throw Error('Assert failed: ' + (message || ''));
};
var minVgi = 0;
var maxVgi = 0.25;
var bins = 10;
function vgi(pixel) {
    var r = pixel[0] / 255;
    var g = pixel[1] / 255;
    var b = pixel[2] / 255;
    return (2 * g - r - b) / (2 * g + r + b);
}
/**
* Summarize values for a histogram.
* @param {numver} value A VGI value.
* @param {Object} counts An object for keeping track of VGI counts.
*/
function summarize(value, counts) {
    var min = counts.min;
    var max = counts.max;
    var num = counts.values.length;
    if (value < min) {
        // do nothing
    } else if (value >= max) {
        counts.values[num - 1] += 1;
    } else {
        var index = Math.floor((value - min) / counts.delta);
        counts.values[index] += 1;
    }
}
function applyBrightness(data, brightness) {
    data[0] += 255 * (brightness / 100);
    data[1] += 255 * (brightness / 100);
    data[2] += 255 * (brightness / 100);
    return data
}
function truncateColor(value) {
    if (value < 0) {
        value = 0;
    } else if (value > 255) {
        value = 255;
    }
    return value;
}
function applyContrast(data, contrast) {
    var factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
    data[0] = truncateColor(factor * (data[0] - 128.0) + 128.0);
    data[1] = truncateColor(factor * (data[1] - 128.0) + 128.0);
    data[2] = truncateColor(factor * (data[2] - 128.0) + 128.0);
    return data
}
function file2url(f) {
    return new Promise((resolve, reject) => {
        var reader = new FileReader();
        reader.onload = function (event) {
            var img = new Image();
            img.onload = function () {
                var canvas = document.createElement('canvas')
                var ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                resolve({
                    url: canvas.toDataURL("image/png"),
                    w: img.width,
                    h: img.height
                })
            }
            img.onerror = function (e) {
                reject('image load error')
            };
            img.src = event.target.result;
        }
        reader.onerror = function (event) {
            reader.abort();
            reject('reader error')
        };
        reader.readAsDataURL(f);
    })
}
function array2rgba(imageArr, w, h){
    const canvas = document.createElement('canvas');
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext("2d");
    const canvas_img = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const canvas_img_data = canvas_img.data;
    const count = w*h
    let min = Number.POSITIVE_INFINITY, max= Number.NEGATIVE_INFINITY;
    const raw = new Uint8Array(imageArr.buffer)
    let type = null
    if(imageArr instanceof Uint8Array){
        for(let i=0;i<count;i++){
            if(imageArr[i]>max) max=imageArr[i]
            if(imageArr[i]<min) min=imageArr[i]
            //encode 16bits to the first two bytes
            canvas_img_data[i*4] = raw[i]
            canvas_img_data[i*4+1] = 0
            canvas_img_data[i*4+2] = 0
            canvas_img_data[i*4+3] = 255
        }
        type = 'uint16'
    }
    else if(imageArr instanceof Uint16Array){
        for(let i=0;i<count;i++){
            if(imageArr[i]>max) max=imageArr[i]
            if(imageArr[i]<min) min=imageArr[i]
            //encode 16bits to the first two bytes
            canvas_img_data[i*4] = raw[i*2]
            canvas_img_data[i*4+1] = raw[i*2+1]
            canvas_img_data[i*4+2] = 0
            canvas_img_data[i*4+3] = 255
        }
        type = 'uint16'
    }
    else if(imageArr instanceof Float32Array){

        for(let i=0;i<count;i++){
            if(imageArr[i]>max) max=imageArr[i]
            if(imageArr[i]<min) min=imageArr[i]
            canvas_img_data[i*4] = raw[i*4]
            canvas_img_data[i*4+1] = raw[i*4+1]
            canvas_img_data[i*4+2] = raw[i*4+2]
            canvas_img_data[i*4+3] = raw[i*4+3]
        }
        type = 'float32'
    }
    else{
        throw "unsupported array type"
    }
    ctx.putImageData(canvas_img, 0, 0);
    return {type: type, url: canvas.toDataURL("image/png"), w:w, h:h, min: min, max: max}
}

function file2img(f) {
    return new Promise(async (resolve, reject) => {
        const decodeimg = ()=>{
            return new Promise((resolve2, reject2) => {
                const reader = new FileReader();
                reader.onload = function (event) {
                    const img = new Image();
                    img.onload = function () {
                        const canvas = document.createElement('canvas')
                        const ctx = canvas.getContext('2d');
                        let min = Number.POSITIVE_INFINITY, max= Number.NEGATIVE_INFINITY;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        resolve2({type: 'rgba', url: canvas.toDataURL("image/png"), w: canvas.width, h: canvas.height, min: 0, max: 255})
                    }
                    img.onerror = function (e) {
                        reject2('image load error')
                    };
                    img.src = event.target.result;
                }
                reader.onerror = function (event) {
                    reader.abort();
                    reject2('reader error')
                };
                reader.readAsDataURL(f);
            })
        }
        if(f.name.endsWith('.tif') || f.name.endsWith('.tiff')){
            const p = await api.getPlugin('Tif File Importer')
            const fileObj = await p.open(f)
            const obj = await fileObj.read()
            console.log(obj.width, obj.height, obj.min, obj.max)
            const ret = array2rgba(obj.array, obj.width, obj.height)
            resolve(ret)
        }
        else if(f.name.endsWith('.png')){
            var reader = new FileReader();
            reader.onload = async function (event) {
                const obj = UPNG.decode(event.target.result)
                if(obj.depth===16 && obj.ctype===0){
                    var area = obj.width * obj.height
                    var nimg = new Uint16Array(area);    // or just  nimg = [];
                    for(var i=0; i<area; i++)  nimg[i] = (obj.data[i*2]<<8) | obj.data[i*2+1] ;
                    const ret = array2rgba(nimg, obj.width, obj.height)
                    resolve(ret)
                }
                else{
                    resolve(await decodeimg())
                }
            }
            reader.onerror = function (event) {
                reader.abort();
                reject('reader error')
            };
            reader.readAsArrayBuffer(f);
        }
        else if(f.name.endsWith('.jpg')){
            resolve(await decodeimg())
        }
        else{
            reject('unsupported file')
        }
    })
}

function fetchDataByUrl(url, type = 'json') {
    return new Promise((resolve, reject) => {
        fetch(url)
            .then((response) => {
                if (type === 'json') {
                    return response.json();
                } else if (type === 'blob') {
                    return response.blob()
                } else {
                    reject('File type error')
                }
            })
            .then((data) => {
                resolve(data)
            })
            .catch((err) => {
                reject(err)
            })
    })
}

const API_VERSION = '0.1.6'
const DATASET_DIR = '/home'
const default_dataset = {
    api_version: API_VERSION,
    version: 1,
    samples: [
        {
            name: '19661_221_G2_1',
            data: {
                "Microtubules": {
                    url: 'https://www.proteinatlas.org/images/19661/221_G2_1_red.jpg',
                    file_name: '19661_221_G2_1_red.jpg'
                },
                "Antibody": {
                    url: 'https://www.proteinatlas.org/images/19661/221_G2_1_green.jpg',
                    file_name: '19661_221_G2_1_green.jpg'
                },
                "Nucleus": {
                    url: 'https://www.proteinatlas.org/images/19661/221_G2_1_blue.jpg',
                    file_name: '19661_221_G2_1_blue.jpg'
                },
                "ER": {
                    url: 'https://www.proteinatlas.org/images/19661/221_G2_1_yellow.jpg',
                    file_name: '19661_221_G2_1_yellow.jpg'
                }
            },
            group: 'train'
        },
        {
            name: '19663_395_C4_2',
            data: {
                "Microtubules": {
                    url: 'https://www.proteinatlas.org/images/19663/395_C4_2_red.jpg',
                    file_name: '19663_395_C4_2_red.jpg'
                },
                "Antibody": {
                    url: 'https://www.proteinatlas.org/images/19663/395_C4_2_green.jpg',
                    file_name: '19663_395_C4_2_green.jpg'
                },
                "Nucleus": {
                    url: 'https://www.proteinatlas.org/images/19663/395_C4_2_blue.jpg',
                    file_name: '19663_395_C4_2_blue.jpg'
                },
                "ER": {
                    url: 'https://www.proteinatlas.org/images/19663/395_C4_2_yellow.jpg',
                    file_name: '19663_395_C4_2_yellow.jpg'
                }
            },
            group: 'train'
        }
    ],
    name: 'default',
    root_folder: '/home/default',
    channel_config: {
        "Microtubules": {
            'filter': '_red.jpg',
            'name': 'Microtubules'
        },
        "Antibody": {
            'filter': '_green.jpg',
            'name': 'Antibody'
        },
        "Nucleus": {
            'filter': '_blue.jpg',
            'name': 'Nucleus'
        },
        "ER": {
            'filter': '_yellow.jpg',
            'name': 'ER'
        }
    }
}
const app = new Vue({
    el: '#app',
    data: {
        COLORS: [
            '#ff0000', '#f44336', '#e91e63', '#9c27b0', '#673ab7',
            '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
            '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
            '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#d500f9',
            '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
            '#00e676', '#64ffda', '#18ffff', '#3f51b5', '#e91e63'
        ],
        gray_slider: null,
        rgb_sliders: null,
        raster: null,
        image_layer: null,
        selected_dataset: null,
        channel_config: null,
        selected_samples: null,
        selected_sample: null,
        selected_channel: null,
        selected_rgb: null,
        selection_dialog: null,
        new_annotation_type: {
            label: 'default',
            color: '#ff0000',
            line_width: 4,
            type: 'Polygon',
            freehand: true
        },
        annotation_types: null,
        selected_annotation: null,
        dataset_list: [],
        selected_dataset_name: '',
        vector_layer: null,
        vector_source: null,
        importSampleCompleteCallback: null,
        bbox: null,
        new_label: '',
        predict_sample: null,
        dataset_folder: 'dataset/',
        default_engine: 'http://127.0.0.1:9527',
        iOS: false,
        message: '',
        checkModalFlag: false,
        loading: false,
        draw_feature_list: [],
        undo_button_flag: true,
        button_list: []
    },
    mounted() {
        this.iOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        const initSlider = (elem)=>{
            noUiSlider.create(elem, {
                start: [0, 255],
                connect: true,
                range: {
                    'min': 0,
                    'max': 255
                },
                tooltips: true,
                format: {
                    to: function (value) {
                        return parseFloat(value).toFixed(0);
                    },
                    from: function (value) {
                        return parseFloat(value).toFixed(0);
                    }
                },
            });
            return elem.noUiSlider
        }
        this.gray_slider = initSlider(this.$refs.gray_slider)
        this.rgb_sliders = [initSlider(this.$refs.r_slider), initSlider(this.$refs.g_slider), initSlider(this.$refs.b_slider)]
        this.initMap()
    },
    methods: {
        async buttonClickCallback (callback) {
            await callback()
        },
        async initDataset() {
            try{
                let path = pathJoin(DATASET_DIR, 'default', 'config.json')
                const data = await readFile(path)
                let dataset = JSON.parse(data)
                if(dataset.api_version !== API_VERSION){
                    api.showMessage('WARNING: Loadding data with a different API version: '+ dataset.api_version)
                }
                assert(default_dataset.version === dataset.version)
                await this.loadDataset(dataset)

            }
            catch(e){
                api.showMessage('loading default dataset.')
                await this.importDataset(default_dataset)
                await this.loadDataset(default_dataset)
            }
        },
        async predict() {
            console.log('this.dataset: ', this.selected_dataset)
            if (!this.selected_sample) {
                api.alert('Please select the sample!')
                return false
            }
            this.predict_sample = this.selected_sample
            api.showStatus('Predicting...')
            api.showMessage('Start predict', 3)
            let path = pathJoin(this.dataset_folder + this.selected_dataset.name, this.selected_sample.group, this.selected_sample.name)
            console.log('path: ', path)
            let result = await api.call('Anet-Lite', 'auto_test', {samples: [path]})
            console.log('predict result: ', result)
            api.showStatus('Finished!')
            api.showMessage('Predict Finished!', 3)
            if (this.predict_sample.name === this.selected_sample.name) {
                if (result) {
                    this.setFeatures(result)
                }
            }
        },
        async setSampleLable(name, label){
            this.selected_sample.labels = this.selected_sample.labels || {}
            this.selected_sample.labels[name] = label
            await this.saveConfig()
            this.$forceUpdate()
        },
        async loadSamples(dataset_name) {
            let path = pathJoin(DATASET_DIR, dataset_name, 'config.json')
            let isExist = await exists(path)
            if (isExist) {
                const data = await readFile(path)
                let dataset = JSON.parse(data)
                if(dataset.api_version !== API_VERSION){
                    api.showMessage('WARNING: Loadding data with a different API version: '+ dataset.api_version)
                }
                console.log('loading dataset ', dataset)
                await this.loadDataset(dataset)
            } else {
                api.alert('No such file: ' + path)
            }
        },
        async datasetSelected(dataset_name) {
            this.selected_dataset_name = dataset_name
        },
        async checkboxChange(annotation) {
            this.updateFeatureStyle()
        },
        convertColor(hex, opacity) {
            let h = hex.replace('#', '');
            let r = parseInt(h.substring(0, 2), 16);
            let g = parseInt(h.substring(2, 4), 16);
            let b = parseInt(h.substring(4, 6), 16);
            return 'rgba(' + r + ',' + g + ',' + b + ',' + opacity / 100 + ')';
        },
        async setAnnotationCheckbox(flag) {
            for (let t in this.annotation_types) {
                this.annotation_types[t].checked = flag
            }
        },
        async showSampleAnnotation() {
            console.log('this.selected_sample: ', this.selected_sample)
            this.draw_feature_list = []

            this.selected_sample.data['annotation.json'] = this.selected_sample.data['annotation.json'] || {'file_name': 'annotation.json'}
            const annotation_data = this.selected_sample.data['annotation.json']
            let annotation_path  = null
             if (this.selected_sample.group === 'test') {
                let prediction = this.selected_sample.data['prediction.json'] || {'file_name': 'prediction.json'}
                annotation_path = pathJoin(this.selected_dataset.root_folder, this.selected_sample.group, this.selected_sample.name, prediction.file_name)
            }
            else{
                annotation_path = pathJoin(this.selected_dataset.root_folder, this.selected_sample.group, this.selected_sample.name, this.selected_sample.data['annotation.json'].file_name)
            }

            if( annotation_path  && await exists(annotation_path)){
                console.log('loading from browserFS: ', annotation_path)
                const data = await readFile(annotation_path)
                const geojsonObject = JSON.parse(data)
                console.log('geojsonObject from browser: ', geojsonObject, annotation_path)
                this.setFeatures(geojsonObject)
            }
            else if(annotation_data.file && annotation_data.file.name){
                const geojsonObject = JSON.parse(await file2text(annotation_data.file))
                console.log('geojsonObject from file: ', geojsonObject, annotation_data.file.webkitRelativePath)
                this.setFeatures(geojsonObject)
            }
            else if(annotation_data.url){
                file = await fetchDataByUrl(annotation_data.url, 'blob')
                const geojsonObject = JSON.parse(await file2text(file))
                console.log('geojsonObject from url: ', geojsonObject, annotation_data.url)
                this.setFeatures(geojsonObject)
            }
            else{
                console.log('No annotation found for ' + this.selected_sample.name)
            }
        },
        async setFeatures(geojsonObject) {
            let features = (new GeoJSON()).readFeatures(geojsonObject)

            this.draw_feature_list = [...this.draw_feature_list, ...features]
            this.undo_button_flag = true

            this.vector_source.addFeatures(features)
            this.updateFeatureStyle()
        },
        async uploadFile(file, path, overwrite, engine) {
            return new Promise(async (resolve, reject) => {
                try {
                    let config = {
                        path: path,
                        engine: engine || this.default_engine,
                        overwrite: overwrite === false ? overwrite : true
                    }
                    let upload_url = await api.requestUploadUrl(config)
                    const fileInfo = await api.uploadFileToUrl({file:file, url: upload_url})
                    console.log(fileInfo.path)
                    resolve(fileInfo)
                } catch (e) {
                    reject(e)
                }
            })
        },
        async checkConfigFile () {

            const engine = await api.call('Anet-Lite', 'get_engine')
            if (engine) {
                this.default_engine = engine
            }
            else{
                throw "cannot determin the engine"
            }

            // upload config.json
            var file = new File([JSON.stringify(this.selected_dataset)], "config.json", {
                type: "text/plain",
            })
            let configPath = this.dataset_folder + this.selected_dataset.name + '/config.json'
            let result = await this.uploadFile(file, configPath, false).catch((err) => {
                console.log('uploadFile err: ', err)
                this.message = 'The configuration file already exists. Do you want to replace it? '
                this.openDialog('modal_file_check')
            })
            if (result) {
                this.train(true)
            }
        },
        async train(overwrite) {
            this.closeDialog('modal_file_check')

            let configPath = this.dataset_folder + this.selected_dataset.name + '/config.json'
            this.uploadDataset(overwrite)
            console.log('configPath: ', configPath)
            const args = {
                configPath: configPath
            }
            api.call('Anet-Lite', 'auto_train', args)
        },
        async typeSelected(type) {
            this.annotation_type = type
        },
        async lineWidthSelected(width) {
            this.line_width = width
        },
        async removeAnnotation(k) {
            if(!this.annotation_types[k]) return
            if (this.selected_annotation === this.annotation_types[k]) {
                this.selected_annotation = null
            }
            delete this.annotation_types[k]
            this.$forceUpdate()
            // clear annotation.json
            let features = this.vector_source.getFeatures()
            let f_list = features.filter((item, index) => {
                if (item.N.label === k) {
                    return item
                }
            })
            f_list.forEach((feature) => {
                this.vector_source.removeFeature(feature);
            })
            let save_list = features.filter((item, index) => {
                if (item.N.label !== k) {
                    return item
                }
            })
            await this.saveAnnotation(save_list)
            this.selected_dataset.annotation_types = this.annotation_types
            await this.saveConfig()
        },
        async closeDialog(id) {
            document.getElementById(id).classList.remove('active')
        },
        async openDialog(id) {
            document.getElementById(id).classList.add('active')
            if (id === 'load_samples') {
                this.dataset_list = await readdir(DATASET_DIR)
            }
        },
        async addAnnotation(annotationSelectedFlag, label) {
            this.new_annotation_type.label = label || this.new_annotation_type.label
            if(!this.new_annotation_type.label){
                return
            }
            if (this.new_annotation_type.label) {
                if(this.annotation_types[this.new_annotation_type.label]){
                    await api.showMessage('Annotation name already exists.')
                    return
                }
                let annot = Object.assign({}, this.new_annotation_type);
                this.new_annotation_type.color = this.COLORS[Math.floor(Math.random() * this.COLORS.length)];
                this.annotation_types[annot.label] = annot
                //this.selected_dataset.annotation_types = this.annotation_types
                if (annotationSelectedFlag) {
                    this.annotationSelected(annot)
                }
                console.log('annotation_types: ', this.annotation_types)
                this.setAnnotationCheckbox(true)
                await this.saveConfig()
            }
        },
        async saveConfig(dataset) {
            dataset = dataset || this.selected_dataset
            const config_path = pathJoin(dataset.root_folder, 'config.json')
            if (!await exists(dataset.root_folder)) {
                await mkdir(dataset.root_folder)
            }
            let stringData = JSON.stringify(dataset)
            await writeFile(config_path, stringData)
        },
        async annotationSelected(annotation) {
            this.selected_annotation = annotation
            annotation.checked = true
            this.switchDrawInteraction()
        },
        async importSamples() {
            this.selection_dialog = await api.showDialog({
                "name": "ImageSelection",
                "type": "ImageSelection",
                "data": {
                    callback: this.importSampleCompleteCallback
                }
            })
        },
        async selectSample(sample) {
            try{
                this.loading = true
                this.selected_sample = sample
                //this.image_layer.setSource()
                this.vector_source.clear(true)
                if(!this.selected_channel || !this.selected_dataset.channel_config[this.selected_channel.name]){
                    if(Object.keys(this.selected_dataset.channel_config).length>0)
                        this.selected_channel = this.selected_dataset.channel_config[Object.keys(this.selected_dataset.channel_config)[0]]
                    else
                        this.selected_channel = null
                }
                if(this.selected_channel){
                    try{
                        if(this.selected_dataset.color_mode){
                            await this.rgbChannelUpdated()
                        }
                        else{
                            await this.channelUpdated(this.selected_channel)
                        }
                    }
                    catch(e){
                        await api.alert(`Channel "${this.selected_channel.name}" does not exist for this sample`)
                        this.clearCanvas()
                    }
                }
                await this.showSampleAnnotation()
            }
            catch(e){
                console.error(e)
            }
            finally{
                this.loading = false
                this.$forceUpdate()
            }

        },
        async getChannelImage(sample, channel_name){
            let dataset = this.selected_dataset
            let data = this.selected_sample.data[channel_name]
            let url = data.url
            if (url) {
                console.log('reading channel image from url', url)
                const imgObj = await url2base64(url)
                imgObj.type = 'rgba'
                imgObj.min = 0
                imgObj.max = 255
                return imgObj
            }
            else if (data.file && data.file.name) {
                console.log('reading channel image from file', data.file.webkitRelativePath)
                return await file2img(data.file)
            }
            else if(data.file_name){
                const file_path = pathJoin(dataset.root_folder, this.selected_sample.group, this.selected_sample.name, data.file_name)

                if(await exists(file_path) && !file_path.endsWith('.base64')){
                     console.log('reading channel image from browser (base64)', file_path)
                    const datab = await readFile(file_path, null)
                    const blob = new Blob([new Uint8Array(datab.buffer)])
                    blob.name = data.file_name
                    return await file2img(blob)
                }
                else if(await exists(file_path)){
                     console.log('reading channel image from browser', file_path)
                    url = await readFile(file_path)
                    const img = await getMeta(url)
                    h = img.height
                    w = img.width
                    return {type: 'rgba', url: url, w: w, h:h, min:0, max: 255}
                }
                else{
                    throw "file not found: " + file_path
                }
            }
            else{
                throw "unable to read the channel image"
            }
        },
        async switchChannelDisplay(color_mode){
            if(color_mode){
                await this.rgbChannelUpdated()
            }
            else{
                const firstChannel = this.selected_dataset.channel_config[Object.keys(this.selected_dataset.channel_config)[0]]
                await this.channelUpdated(firstChannel)
            }
        },
        async rgbChannelUpdated(channel) {
            try{
                this.loading = true
                this.$forceUpdate()
                let view = this.map.getView()
                const zoom = view && view.getZoom()
                const center = view &&  view.getCenter()
                view = null
                this.selected_rgb = this.selected_rgb || []
                if(channel){
                    if (channel.checked) {
                        this.selected_rgb.push(channel)
                        if (this.selected_rgb.length > 3) {
                            this.selected_rgb.shift()
                        }
                    } else {
                        const index = this.selected_rgb.indexOf(channel);
                        if (index > -1) {
                            this.selected_rgb.splice(index, 1);
                        }
                    }
                }
                const image_sources = []
                let projection = null
                let w = 2048,
                    h = 2048;
                let extent_ext =  [0, 0, 0, 0];
                const ranges = []
                const configs = []
                for (let i=0; i<this.selected_rgb.length; i++) {
                    const channel = this.selected_rgb[i]
                    const imgObj = await this.getChannelImage(this.selected_sample, channel.name)
                    configs.push({type: imgObj.type, min: imgObj.min, max: imgObj.max})
                    ranges.push([imgObj.min, imgObj.max])
                    const extent =  [0, 0, imgObj.w, imgObj.h];
                    if(extent_ext[2]<imgObj.w){
                        extent_ext[2] = imgObj.w
                    }
                    if(extent_ext[3]<imgObj.h){
                        extent_ext[3] = imgObj.h
                    }
                    this.rgb_sliders[i].updateOptions({
                        range: {
                            'min': imgObj.min,
                            'max': imgObj.max
                        }
                    });
                    this.rgb_sliders[i].off('change')
                    this.rgb_sliders[i].on('change', ()=> {
                        const range = this.rgb_sliders[i].get();
                        range[0] = parseInt(range[0])
                        range[1] = parseInt(range[1])
                        ranges[i] = range
                        this.raster.set('ranges', ranges)
                        this.raster.changed()
                    });
                    projection = new Projection({
                        code: 'image',
                        units: 'pixels',
                        extent: extent
                    });
                    const image_source = new Static({
                        url: imgObj.url,
                        projection: projection,
                        imageExtent: extent,
                        crossOrigin: 'anonymous'
                    })
                    image_sources.push(image_source)
                    //TODO: autoadjust the range
                    const initRange = [imgObj.min, imgObj.max]
                    ranges[i] = initRange
                    this.rgb_sliders[i].set(initRange);
                }
                this.raster = this.createRasterLayer(image_sources, configs, ranges)
                this.image_layer.setSource(this.raster)
                this.bbox = extent_ext

                this.bbox[2] = this.bbox[2] - 1
                this.bbox[3] = this.bbox[3] - 1

                console.log(this.bbox)

                this.map.setView(new View({
                    projection: projection,
                    center: center || getCenter(extent),
                    extent: extent_ext,
                    zoom: zoom || 2,
                    maxZoom: 8
                }))
                this.raster.set('ranges', ranges)
                this.raster.changed()
            }
            catch(e){
                console.error(e)
                throw e
            }
            finally{
                // this.loading = false
                this.$forceUpdate()
            }

        },
        async channelUpdated(channel) {
            try{
                console.log('channel: ', channel)
                this.loading = true
                let view = this.map.getView()
                const zoom = view && view.getZoom()
                const center = view &&  view.getCenter()
                view = null
                this.selected_channel = channel
                const imgObj = await this.getChannelImage(this.selected_sample, channel.name)
                this.gray_slider.updateOptions({
                    range: {
                        'min': imgObj.min,
                        'max': imgObj.max
                    }
                });
                this.gray_slider.off('change')
                this.gray_slider.on('change', ()=> {
                    const range = this.gray_slider.get();
                    range[0] = parseInt(range[0])
                    range[1] = parseInt(range[1])
                    this.raster.set('ranges', [range]);
                    this.raster.changed()
                });
                let extent = [0, 0, imgObj.w, imgObj.h];
                this.bbox = [0, 0, imgObj.w - 1, imgObj.h - 1];
                let projection = new Projection({
                    code: 'image',
                    units: 'pixels',
                    extent: extent
                })
                const image_source = new Static({
                    url: imgObj.url,
                    projection: projection,
                    imageExtent: extent,
                    crossOrigin: 'anonymous'
                })
                const configs = [{type: imgObj.type, min: imgObj.min, max: imgObj.max}]
                const ranges = [[imgObj.min, imgObj.max]]

                if(this.iOS){
                    this.image_layer.setSource(image_source)
                }
                else{
                    this.raster = this.createRasterLayer([image_source], configs, ranges)
                    this.image_layer.setSource(this.raster)
                    //TODO: autoadjust the range
                    const initRange = [imgObj.min, imgObj.max]
                    this.gray_slider.set(initRange);
                    this.raster.set('ranges', [initRange]);
                    this.raster.changed()
                }

                this.map.setView( new View({
                    projection: projection,
                    center: center || getCenter(extent),
                    extent: extent,
                    zoom: zoom || 2,
                    maxZoom: 8
                }))
            }
            catch(e){
                console.error(e)
                throw e
            }
            finally{
                this.loading = false
                this.$forceUpdate()
            }
        },
        clearCanvas(){
            this.image_layer.setSource(null)
        },
        async importDataset(dataset) {
            try{
                this.loading = true
                const samples = dataset.samples
                if (samples.length <= 0) {
                    api.alert('No sample found in the dataset')
                    return
                }
                let path = pathJoin(DATASET_DIR, dataset.name)
                if (!await exists(path)) await mkdir(path)
                const labels = {}
                for (let i = 0; i < samples.length; i++) {
                    const sample = samples[i]
                    const sample_dir = pathJoin(dataset.root_folder, sample.group, sample.name)
                    await mkdir(sample_dir)
                    for (let k of Object.keys(sample.data)) {
                        // only import channels or annotations/predictions
                        const channels_or_json = dataset.channel_config[k] || sample.data[k].file_name === 'annotation.json' || sample.data[k].file_name === 'annotation.json'
                        if(channels_or_json && sample.data[k].file){
                            let file_path = pathJoin(sample_dir, sample.data[k].file_name)

                            let file = null
                            if (sample.data[k].file && sample.data[k].file.type) {
                                file = sample.data[k].file
                            } else if (sample.data[k].url) {
                                file = await fetchDataByUrl(sample.data[k].url, 'blob')
                            } else {
                                throw 'unsupported data type.'
                            }
                            let arraybuffer = await file2arraybuffer(file)
                            await writeFile(file_path, arraybuffer)
                        }
                    }
                }
                await this.saveConfig(dataset)
            }
            catch(e){
                console.error(e)
                throw e
            }
            finally{
                this.loading = false
                this.$forceUpdate()
            }

        },
        async loadDataset(dataset) {
            try{
                this.selected_dataset = dataset
                this.selected_samples = dataset.samples || []
                this.channel_config = dataset.channel_config
                this.selected_rgb = this.selected_dataset.rgb_channels || []
                dataset.annotation_types = dataset.annotation_types || {}
                this.annotation_types = dataset.annotation_types

                // if(import_annotation){
                //     const labels = await this.importAnnotation(dataset)
                //     for(let label of labels){
                //         if(!this.annotation_types[label]){
                //             await this.addAnnotation(false, label)
                //         }
                //     }
                // }

                if(Object.keys(this.annotation_types).length <= 0){
                    this.addAnnotation(false, 'default')
                }
                await this.stopAnnotation()
                this.setAnnotationCheckbox(true)
                if (this.selected_annotation) this.switchDrawInteraction()
                await this.selectSample(this.selected_samples[0])
            }
            catch(e){
                console.error(e)
                throw e
            }
            finally{
                this.loading = false
            }
        },
        async importAnnotation(dataset){
            const labels = {}
            for (let i = 0; i < dataset.samples.length; i++) {
                const sample = dataset.samples[i]
                const data = sample.data['annotation.json']
                if(!data || !data.file || !data.file.size){
                  continue
                }
                const sample_dir = pathJoin(DATASET_DIR, dataset.name, sample.group, sample.name)
                await mkdir(sample_dir)
                let filePath = pathJoin(sample_dir, data.file.name)
                let jsonStr = await file2text(data.file)
                let annotationObj = JSON.parse(jsonStr)
                let features = annotationObj.features
                for (let item of features) {
                    labels[item.properties.label] = true
                }
                await writeFile(filePath, jsonStr)
            }
            return Object.keys(labels)
        },
        async initMap() {
            const vector_source = new VectorSource({
                wrapX: false
            });
            const vector = new VectorLayer({
                source: vector_source
            });
            const select = new Select({
                wrapX: false
            });
            // const modify = new Modify({
            //     source: vector_source,
            //     features: select.getFeatures(),
            //     deleteCondition: function (event) {
            //         return ol.events.condition.shiftKeyOnly(event) &&
            //             ol.events.condition.singleClick(event);
            //     }
            // });
            // modify.createVertices = true;
            this.raster = this.createRasterLayer([], [], [])
            this.image_layer = new ImageLayer({
                source: this.raster
            })
            const map = new Map({
                layers: [
                    this.image_layer,
                    vector
                ],
                target: 'map',
                view: new View({
                    //projection: projection,
                    //center: getCenter(extent),
                    zoom: 2,
                    maxZoom: 8
                })
            });
            //map.addControl(new LayerSwitcher());
            map.addControl(new MousePosition({
                coordinateFormat: createStringXY(0)
            }));
            //map.addInteraction(modify);
            // map.addInteraction(new DragAndDrop({
            //     source: vector_source,
            //     formatConstructors: [GeoJSON]
            // }));
            map.addInteraction(select);
            select.on('select', (e) => {
                console.log(e.selected, '&nbsp;' +
                    e.target.getFeatures().getLength() +
                    ' selected features (last operation selected ' + e.selected.length +
                    ' and deselected ' + e.deselected.length + ' features)')
            });
            this.map = map;
            this.vector_layer = vector;
            this.vector_source = vector_source;
            this.select = select;
        },
        createRasterLayer(sources, configs, ranges) {
            const mkOperation = ()=>{
                for(let c of configs){
                    assert(c.type==='rgba'||c.type==='uint16'||c.type==='float32' , 'Unsupported type: '+ c)
                }
                if(sources.length===1){
                    if(configs[0].type === 'rgba'){
                        return function (pixels, data) {
                            const pixel = pixels[0];
                            const range = data.ranges[0]
                            pixel[0] = (pixel[0]-range[0])/(range[1]-range[0])*255;
                            pixel[1] = (pixel[1]-range[0])/(range[1]-range[0])*255;
                            pixel[2] = (pixel[2]-range[0])/(range[1]-range[0])*255;
                            pixel[3] = 255;
                            return pixel
                        }
                    }
                    else if(configs[0].type === 'uint16'){
                        return function (pixels, data) {
                            const pixel = pixels[0];
                            const range = data.ranges[0]
                            const raw = new Uint8Array(new ArrayBuffer(4));
                            raw[0] = pixel[0]
                            raw[1] = pixel[1]
                            const v = new Uint16Array(raw.buffer)[0]
                            pixel[0] = truncateColor((v-range[0])/(range[1]-range[0])*255);
                            pixel[1] = pixel[0];
                            pixel[2] = pixel[0];
                            pixel[3] = 255;
                            return pixel
                        }
                    }
                    else if(configs[0].type === 'float32'){
                        return function (pixels, data) {
                            const pixel = pixels[0];
                            const range = data.ranges[0]
                            const raw = new Uint8Array(new ArrayBuffer(4));
                            raw[0] = pixel[0]
                            raw[1] = pixel[1]
                            raw[2] = pixel[2]
                            raw[3] = pixel[3]
                            const v = new Float32Array(raw.buffer)[0]
                            pixel[0] = truncateColor((v-range[0])/(range[1]-range[0])*255);
                            pixel[1] = pixel[0];
                            pixel[2] = pixel[0];
                            pixel[3] = 255;
                            return pixel
                        }
                    }
                }
                else if(sources.length>1){
                    return function (pixels, data) {
                        const ret = [0,0,0,255]
                        let v = null
                        for(let i=0;i<pixels.length;i++){
                            const pixel = pixels[i];
                            const range = data.ranges[i]
                            if(data.configs[i].type === 'rgba'){
                                v =  truncateColor(1.0*((pixel[0]+pixel[1]+pixel[2])/3.0-range[0])/(range[1]-range[0])*255);
                                ret[i] = v
                            }
                            else if(data.configs[i].type === 'uint16'){
                                const raw = new Uint8Array(new ArrayBuffer(4));
                                raw[0] = pixel[0]
                                raw[1] = pixel[1]
                                let v = new Uint16Array(raw.buffer)[0]
                                v = truncateColor(1.0*(v-range[0])/(range[1]-range[0])*255);
                                ret[i] = v
                            }
                            else if(data.configs[i].type === 'float32'){
                                const raw = new Uint8Array(new ArrayBuffer(4));
                                raw[0] = pixel[0]
                                raw[1] = pixel[1]
                                raw[2] = pixel[2]
                                raw[3] = pixel[3]
                                let v = new Float32Array(raw.buffer)[0]
                                v =  truncateColor(1.0* (v-range[0])/(range[1]-range[0])*255);
                                ret[i] = v
                            }
                        }
                        return ret
                    }
                }
            }
            const raster = new RasterSource({
                sources: sources,
                operation: mkOperation(),
                lib: {
                    truncateColor: truncateColor,
                }
            });
            raster.set('ranges', ranges);
            raster.set('configs', configs);
            raster.on('beforeoperations', function (event) {
                event.data.ranges = raster.get('ranges');
                event.data.configs = raster.get('configs');
            });
            raster.on('afteroperations', function (event) {});
            return raster
        },
        switchDrawInteraction(type, freehand) {
            if(!this.selected_annotation) return
            type = type || this.selected_annotation.type
            if(type==='Label'){
                if (this.draw) {
                    this.map.removeInteraction(this.draw)
                }
                return
            }
            freehand = freehand || this.selected_annotation.freehand
            if(freehand === undefined) freehand = true
            if (this.draw) {
                this.map.removeInteraction(this.draw)
            }
            const draw = new Draw({
                source: this.vector_source,
                type: type,
                freehand: freehand,
                style: new Style({
                    fill: new Fill({
                        color: 'rgba(255, 255, 255, 0.2)'
                    }),
                    stroke: new Stroke({
                        color: this.selected_annotation.color,
                        width: this.selected_annotation.line_width
                    })
                })
            });
            this.map.addInteraction(draw);
            draw.on('drawend', async (evt) => {
                const feature = evt.feature;
                if(!this.selected_annotation) return
                if(this.selected_annotation.label === 'default'){
                    const new_name = await api.prompt("Please set a name for the annotation your are making!")

                    if(new_name && new_name !== 'default'){
                        this.selected_annotation.label = new_name
                        this.annotation_types[new_name] = this.annotation_types['default']
                        delete this.annotation_types['default']
                        this.$forceUpdate()
                    }
                }
                feature.set('label', this.selected_annotation.label)
                this.draw_feature_list.push(feature)
                this.undo_button_flag = true
                await this.updateFeatureStyle()
                await this.saveAnnotation()
            });
            this.draw = draw
        },
        async saveAnnotation(features){
            const allFeatures = features || this.vector_source.getFeatures();
            let sample = this.selected_sample
            let sample_path = pathJoin(this.selected_dataset.root_folder, sample.group, sample.name)
            let annotationPath = null
            if (sample.data['annotation.json'] && sample.data['annotation.json'].file_name) {
                annotationPath = pathJoin(sample_path, sample.data['annotation.json'].file_name)
            } else {
                annotationPath = pathJoin(sample_path, 'annotation.json')
                sample.data['annotation.json'] = {file_name: 'annotation.json'}
            }
            if (!await exists(sample_path)) {
                await mkdir(sample_path)
            }
            const format = new ol.format.GeoJSON();
            const routeFeatures = format.writeFeaturesObject(allFeatures, {decimals: 1});
            routeFeatures['bbox'] = this.bbox
            await writeFile(annotationPath, JSON.stringify(routeFeatures))
            await this.saveConfig()
        },
        // Save annotations
        exportAnnotation() {
            const allFeatures = this.vector_source.getFeatures();

            const format = new ol.format.GeoJSON();
            const routeFeatures = format.writeFeaturesObject(allFeatures, {decimals: 1});
            routeFeatures['bbox'] = this.bbox
            const blob = new Blob([JSON.stringify(routeFeatures)], {
                type: "text/plain;charset=utf-8"
            });
            const name_save = this.selected_sample.name + '_annotation.json'
            api.exportFile(blob, name_save)
        },
        async exportAllAnnotation(with_images){
            const zip = new JSZip()
            const allFeatures = this.vector_source.getFeatures();

            const format = new ol.format.GeoJSON();
            const routeFeatures = format.writeFeaturesObject(allFeatures, {decimals: 1});
            routeFeatures['bbox'] = this.bbox
            const blob = new Blob([JSON.stringify(routeFeatures)], {
                type: "text/plain;charset=utf-8"
            });


            const samples = this.selected_samples
            const dataset = this.selected_dataset
            if (samples.length <= 0) {
                api.alert('No sample found in the dataset')
                return
            }

            const labels = {}
            api.showMessage('Collecting data ...')
            await this.walkThroughSamples(dataset, samples, (sample, data, file)=>{
                zip.file(pathJoin(dataset.name, sample.group, sample.name, data.file_name), file)
            }, with_images)
            api.showMessage('Making zip archive...')
            const zipBlob = await zip.generateAsync({ type:"blob",
                                compression: "DEFLATE",
                                compressionOptions: {
                                    level: 9
                                }},
                                (mdata)=>{
                                    api.showProgress(mdata.percent)
                                }
            )
            api.showMessage('Dataset '+dataset.name+' exported successfully.')
            const id =  randId();
            const zip_file_name = dataset.name +'-'+ id + '_exported.zip'
            const zipFile = new File([zipBlob], zip_file_name, {type: "application/zip", lastModified: Date.now()});
            console.log(zipFile)
            api.exportFile(zipFile, zip_file_name)
        },
        async walkThroughSamples(dataset, samples, run_callback, with_images){

            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i]
                const sample_dir = pathJoin(dataset.root_folder, sample.group, sample.name)
                if(with_images){
                    for (let k of Object.keys(sample.data)) {
                        // only export channels
                        if(dataset.channel_config[k]){

                            if(sample.data[k].file){
                                console.log('adding file to archive ',sample.data[k].file_name)
                                run_callback(sample, sample.data[k],  sample.data[k].file)
                            }
                            else if(sample.data[k].url){
                                console.log('adding url data to archive ',sample.data[k].file_name)
                                const file = await fetchDataByUrl(sample.data[k].url, 'blob')
                                run_callback(sample, sample.data[k], new Blob([file]))
                            }
                        }
                        else{
                            console.log('skip ', sample.data[k])
                        }
                    }
                }

                for(let file_name of ['annotation.json', 'prediction.json']){
                    let file_path = pathJoin(sample_dir, file_name)
                    if (await exists(file_path)){
                        const datab = await readFile(file_path, null)
                        const blob = new Blob([new Uint8Array(datab.buffer)])
                        const file = new File([blob], file_name, {type: "application/octet-stream", lastModified: Date.now()});
                        console.log('adding local file: '+sample.name + '/' + file_name)
                        run_callback(sample, sample.data[file_name], file)
                    }
                    else if(sample.data[file_name] && sample.data[file_name].url){
                        const file = await fetchDataByUrl(sample.data[file_name].url, 'blob')
                        console.log('adding remote file: '+sample.name + '/' + file_name)
                        run_callback(sample, sample.data[file_name], new Blob([file]))
                    }
                    else{
                        console.log(sample.name + '/' + file_name + ' not found.')
                    }
                }
            }
        },
        async uploadAnnotation() {
            const targetObj = await api.showFileDialog({type: 'directory', mode: 'single'})
            const allFeatures = this.vector_source.getFeatures();

            const format = new ol.format.GeoJSON();
            const routeFeatures = format.writeFeaturesObject(allFeatures, {decimals: 1});
            routeFeatures['bbox'] = this.bbox

            const blob = new Blob([JSON.stringify(routeFeatures)], {
                type: "text/plain;charset=utf-8"
            });

            var file = new File([blob], "config.json", {
                type: "text/plain",
            })
            console.log('uploading to ',  targetObj.engine, targetObj.path)
            await this.uploadFile(file, pathJoin(targetObj.path, 'annotation.json'), true, targetObj.engine)

        },
        async uploadAllAnnotations(with_images) {
            const targetObj = await api.showFileDialog({type: 'directory', mode: 'single'})
            console.log('uploading dataset: ', this.selected_dataset, ' to ', targetObj.engine, targetObj.path)
            try{
                await this.uploadDataset(targetObj, with_images, false)
            }
            catch(e){
                console.error(e)
                const ret = await api.confirm({content: 'Error occured, would you like to overwrite the dataset?', confirm_text: 'Yes, overwrite'})
                if (ret) {
                    //TODO: fix this
                    await this.uploadDataset(targetObj, with_images, true)
                }
            }
        },
        async uploadDataset(targetObj, with_images, overwrite){

            api.showMessage('Uploading config file...')
            // upload config.json
            var file = new File([JSON.stringify(this.selected_dataset)], "config.json", {
                type: "text/plain",
            })
            const targetDir = pathJoin(targetObj.path, this.selected_dataset.name)
            await this.uploadFile(file, pathJoin(targetDir, 'config.json'), overwrite, targetObj.engine)

            api.showMessage('Uploading dataset: ', this.selected_dataset, ' to ', targetObj.engine, targetObj.path)
            await this.walkThroughSamples(this.selected_dataset, this.selected_dataset.samples, (sample, data, file)=>{
                this.uploadFile(file, pathJoin(targetDir, sample.group, sample.name, data.file_name), overwrite, targetObj.engine)
            }, with_images)
            api.showMessage('The dataset is successfully uploaded')
        },
        // Clear window
        async deleteAnnotation() {
            if (this.vector_source) {
                const features = this.select.getFeatures()
                if(features.getLength()>0){
                    features.forEach((feature)=>{
                        console.log('removing ', feature)
                        this.vector_source.removeFeature(feature);
                        this.select.getFeatures().remove(feature);
                    });
                }
                else{
                    try {
                        const answer = await api.confirm({content: "Do you really want to delete all the visible annotations?", confirm_text: 'Yes, delete all.'})
                        if (answer) {
                            const features = this.vector_source.getFeatures();
                            features.forEach((feature) => {
                                const ann = this.annotation_types[feature.get('label')]
                                if(ann && ann.checked){
                                    this.vector_source.removeFeature(feature);
                                }
                            });
                            this.undo_button_flag = false
                        }
                    } catch (err) {
                        return;
                    }
                }
            }
        },
        async stopAnnotation() {
            this.selected_annotation = null
            if(this.draw) this.map.removeInteraction(this.draw)
        },
        // Undo last annotation
        undoDraw() {
            const features = this.draw_feature_list;
            if (features.length > 0) {
                this.vector_source.removeFeature(features[features.length - 1]);
                this.draw_feature_list.pop()
                if (this.draw_feature_list.length === 0) {
                    this.undo_button_flag = false
                }
            } else {
                api.showMessage('The feature list is empty.')
            }
        },
        addNewLabel(){
            this.new_annotation_type.labels = this.new_annotation_type.labels ||[]
            if(this.new_annotation_type.labels.indexOf(this.new_label)>-1){
                api.showMessage('label already exists.')
                return
            }

            this.new_annotation_type.labels.push(this.new_label)
            this.new_label=''
        },
        removeLabel(label){
            const index = this.new_annotation_type.labels.indexOf(label)
            if( index>-1)
            this.new_annotation_type.labels.splice(index, 1)
            this.$forceUpdate()
        },
        async updateFeatureStyle() {
            this.vector_layer.setStyle((feature) => {
                const label = feature.get('label')
                let annotation = this.annotation_types[label]
                if(!annotation && label){
                    this.addAnnotation(false, label)
                }
                if(annotation){
                    if (annotation.checked) {
                        let color_style = new Style({
                            fill: new Fill({
                                color: 'rgba(255, 255, 255, 0.2)'
                            }),
                            stroke: new Stroke({
                                color: annotation.color,
                                width: annotation.line_width
                            }),
                            text: new Text({
                                text: label,
                                font: '14px Calibri,sans-serif',
                                fill: new Fill({
                                    color: '#000'
                                }),
                                stroke: new Stroke({
                                    color: '#fff',
                                    width: 4
                                })
                            })
                        });
                        return color_style;
                    }
                    else{
                        return new Style({});
                    }
                }
            })
        }
    }
})
class ImJoyPlugin {
    async setup() {
        app.importSampleCompleteCallback = this.importSampleComplete
        app.getData = this.getData
        app.getDataset = this.getDataset
        app.plugin = this
    }
    async getDataset(){
        const filerted_dataset = {}
        for(let k in app.selected_dataset){
            if(k!=='samples')
                filerted_dataset[k] = app.selected_dataset[k]
            else{
                const samples = app.selected_dataset[k]
                const filtered_samples = []
                for(let sample of samples){
                    const filtered_sample = {name: sample.name, data: {}, group: sample.group}
                    for(let d in sample.data){
                        filtered_sample.data[d] = {file_name: sample.data[d].file_name}
                    }
                    filtered_samples.push(filtered_sample)
                }
                filerted_dataset[k] = filtered_samples
            }
        }
        return filerted_dataset
    }
    async getData(dataset_name, sample_name, data_key){
        assert(app.selected_dataset.name === dataset_name)
        let samples = app.selected_dataset.samples.filter(s=>s.name===sample_name)
        if(samples.length>0){
            const sample = samples[0]
            const data = sample.data[data_key]
            if(!data){
                throw "Data not found"
            }
            else{
                if(app.selected_dataset.channel_config[data_key]){
                    return app.getChannelImage(sample, data_key)
                }
                else if(data.file) {
                    return await file2base64(data.file)
                }
                else{
                    const sample_path = pathJoin(this.selected_dataset.root_folder, sample.group, sample.name)
                    const browser_path = pathJoin(sample_path, sample.data[data_key].file_name)
                    if(await exists(browser_path)){
                        content = await readFile(browser_path)
                        return content
                    }
                }
                console.error(data)
                throw "unsupported data type"
            }
        }
        else{
            throw "Sample not found"
        }
    }
    async importSampleComplete(dataset, save) {
        app.selection_dialog.close()
        if(!dataset.root_folder.startsWith('/')){
            dataset.root_folder = pathJoin(DATASET_DIR, dataset.root_folder)
        }
        if(save){
            assert(dataset.api_version === API_VERSION, 'dataset API version mismatch')
            await app.importDataset(dataset)
            await app.loadDataset(dataset)
        }
        else{
            dataset.root_folder = pathJoin('/tmp', dataset.name, randId())
            await app.loadDataset(dataset)
        }
    }

    async loadEngineData(dataset) {
        let root_url = await api.getFileUrl({'path': dataset.root_folder, 'engine': dataset.engine_url})
        let root_items = await fetchDataByUrl(root_url)
        root_items = root_items.filter((item) => {
            return item.name && !item.name.startsWith('.') && !(item.name.startsWith('__'))
        })

        let dataset_name = dataset.name
        let channel_config = {}
        let arr = root_items.filter((item) => { return item.type === 'file' && item.name === 'config.json' })
        if (arr && arr.length > 0) {
            let config_file_url = root_url + '/' + arr[0].name
            let configData = await fetchDataByUrl(config_file_url)
            channel_config = configData.channel_config || {}
            dataset.annotation_types = configData.annotation_types || {}
        } else {
            channel_config = dataset.channel_config
        }
        if (!dataset.samples) {
            dataset.samples = []
        }
        const get_sample = async (item) => {
            if (item.type === 'dir') {
                let sample_folder = await fetchDataByUrl(root_url + '/' + item.name)
                for (let sf of sample_folder) {
                    let sample = {}
                    let data = {}
                    if (sf.type === 'dir') {
                        let files = await fetchDataByUrl(root_url + '/' + item.name + '/' + sf.name)
                        for (let f of files) {
                            let name = f.name
                            data[name] = {
                                file_name: f.name,
                                path: dataset_name + '/' + item.name + '/' + sf.name + '/' + f.name,
                                url: root_url + '/' + item.name + '/' + sf.name + '/' + f.name
                            }
                        }
                    }

                    let dataSort = {}
                    Object.keys(data).sort().forEach((k) => {
                        dataSort[k] = data[k]
                    })
                    sample['data'] = dataSort
                    sample['name'] = sf.name
                    sample['group'] = item.name
                    dataset.samples.push(sample)
                }
            }
        }
        for(let item of root_items){
            await get_sample(item)
        }

        for (let c in channel_config) {
            let channelName = c
            let filter = channel_config[c].filter

            let annotationFlag = 0
            for(let sample of dataset.samples){
                for(let fn in sample.data){
                    if(sample.data[fn].file_name.indexOf(filter) >-1){
                        if (sample.data[fn].file_name === 'annotation.json' || sample.data[fn].file_name === 'prediction.json') {
                            api.showMessage('The annotation files cannot be used as channel')
                            annotationFlag = 1
                        } else {
                            sample.data[channelName] = sample.data[fn]
                            if (fn !== channelName) {
                                delete sample.data[fn]
                            }
                        }
                    } else {
                        console.log('No match to file. ')
                    }
                }
            }
            if (annotationFlag === 0) {
                channel_config[channelName] = {filter: this.filter, name: channelName}
            }
        }

        dataset.channel_config = channel_config
        console.log(JSON.stringify(dataset))
        await app.loadDataset(dataset)
    }

    // Run function
    async run(my) {
        console.log(my)

        if (my.data && my.data.dataset) {
            if (my.data.actions && my.data.actions.length > 0) {
                app.button_list = my.data.actions
            }
            this.loadEngineData(my.data.dataset)
        } else {
            app.initDataset()
        }
    }
}
api.export(new ImJoyPlugin())
</script>
<window lang="html">
    <div id="app">
        <div class="loading loading-lg overlay" v-show="loading"></div>
        <div class="dropdown">
            <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                <i class="fas fa-folder-plus icon-size-position"></i>
                <label class="hide-sm">File</label>  <i class="icon icon-caret"></i>
            </a>
            <ul class="menu">
                <li v-if="selected_samples" v-for="sample in selected_samples" :key="sample.name" class="menu-item"
                    @click="selectSample(sample)">
                    <a href="#" :class="selected_sample===sample?'active': ''">
                        {{sample.group}} / {{sample.name}}
                    </a>
                </li>
                <li>
                    <div style="display:flex;">
                        <a href="#">
                            <button class="btn btn-primary" @click="importSamples">Import Samples</button>
                        </a>&nbsp;&nbsp;
                        <a href="#">
                            <button class="btn btn-primary" @click="openDialog('load_samples')">Load</button>
                        </a>
                    </div>
                </li>
            </ul>
        </div>
        <div class="dropdown">
            <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                <i class="fas fa-layer-group icon-size-position"></i>
                <label class="hide-sm">Channel</label>  <i class="icon icon-caret"></i>
            </a>
            <ul class="menu" v-if="selected_dataset && selected_dataset.channel_config">
                <li v-if="!iOS">
                    <div class="form-group">
                        <label class="form-switch">
                            <input type="checkbox" v-model="selected_dataset.color_mode" @change="switchChannelDisplay(selected_dataset.color_mode)">
                            <i class="form-icon"></i> Color Mode
                        </label>
                    </div>
                </li>
                <li v-for="(channel, k) in selected_dataset.channel_config" :key="k" class="menu-item cursor-pointer">
                    <label class="form-checkbox" v-if="selected_dataset.color_mode">
                        <input type="checkbox" v-model="channel.checked" @change="rgbChannelUpdated(channel)">
                        <i class="form-icon"></i> {{channel.name}}
                    </label>
                    <a href="#" :class="selected_channel===channel?'active': ''" v-else @click="channelUpdated(channel)">
                        <i class="icon icon-link"></i> {{channel.name}}
                    </a>
                </li>
            </ul>
        </div>
        <div class="dropdown">
            <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                <i class="fas fa-user-edit icon-size-position"></i>
                <label class="hide-sm">Annotation&nbsp;</label>
                <i class="icon" :class="selected_annotation?'':'icon-caret'" :style="{ backgroundColor: selected_annotation?convertColor(selected_annotation.color, 80):'rgb(255,255,255)' }"></i>
            </a>
            <ul class="menu">
                <li v-if="selected_annotation"  class="menu-item cursor-pointer">
                    <a href="#"  @click="stopAnnotation()">
                       <i class="icon icon-stop"></i> Stop Annotation
                    </a>
                </li>
                <li v-if="selected_annotation"  class="menu-item cursor-pointer">
                    <label class="form-switch">
                        <input v-model="selected_annotation.freehand" @change="switchDrawInteraction()"
                            type="checkbox">
                        <i class="form-icon"></i>Free Hand
                    </label>
                </li>
                <li v-for="(an, label) in annotation_types"  style="display:flex;"
                    :style="selected_annotation && selected_annotation.label === label? 'background:#cde1fd;':''"
                        class="menu-item cursor-pointer" :key="label">
                     <label class="form-checkbox">
                        <input type="checkbox" v-model="an.checked" @change="checkboxChange(an)">
                        <i class="form-icon" :style="{backgroundColor: an.color}"></i>
                    </label>
                    <span class="cursor" style="padding: .25rem .4rem;" @click.stop="annotationSelected(an)">{{an.label}}</span>
                </li>
                <li  class="menu-item cursor-pointer">
                    <a href="#">
                        <button class="btn btn-primary" @click="openDialog('annotation_dialog')">
                            <i class="icon icon-plus"></i>&nbsp;New Marker
                        </button>
                    </a>
                </li>
            </ul>
        </div>

        <button class="btn btn-sm" title="delete" @Click="deleteAnnotation()"><i class="icon icon-delete"></i></button>
        <button :disabled="!undo_button_flag" class="btn btn-sm" title="undo" @click="undoDraw()"><i class="fas fa-undo"></i></button>

        <div id="buttons" class="btn-group">
            <div class="dropdown">
                <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                    <i class="fas fa-upload icon-size-position"></i>
                    <label class="hide-sm">Upload</label>
                    <i class="icon icon-caret"></i>
                </a>
                <ul class="menu" style="width: 200px;">
                    <li class="cursor-pointer"
                        @click="uploadAnnotation()">Current annotation</li>
                     <li class="cursor-pointer"
                        @click="uploadAllAnnotations(false)">All Annotations</li>
                     <li class="cursor-pointer"
                        @click="uploadAllAnnotations(true)">Channels + Annotations</li>
                </ul>
            </div>
            &nbsp;
            <div class="dropdown">
                <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                    <i class="fas fa-download icon-size-position"></i>
                    <label class="hide-sm">Export</label>
                    <i class="icon icon-caret"></i>
                </a>
                <ul class="menu" style="width: 200px;">
                    <li class="cursor-pointer"
                        @click="exportAnnotation()">Current annotation</li>
                     <li class="cursor-pointer"
                        @click="exportAllAnnotation(false)">All Annotations</li>
                     <li class="cursor-pointer"
                        @click="exportAllAnnotation(true)">Channels + Annotations</li>
                </ul>
            </div>
        </div>
        <div id="buttons" class="btn-group">
            <button v-for="v in button_list" :title="v.tooltip" class="btn btn-sm" @click="buttonClickCallback(v.callback)">{{v.name}}</button>
        </div>
        <!-- <div id="buttons" class="btn-group">
            <button class="btn btn-sm" @click="checkConfigFile()">Train</button>
        </div>
        <div id="buttons" class="btn-group">
            <button class="btn btn-sm" @click="predict">Predict</button>
        </div> -->
        <div id="buttons" class="btn-group" v-if="selected_annotation && selected_annotation.type==='Label'">
            <button class="btn btn-sm" v-for="label in selected_annotation.labels" :key="label" :class="selected_sample.labels && selected_sample.labels[selected_annotation.label] === label?'btn-primary': ''" @click="setSampleLable(selected_annotation.label, label)">{{label}}</button>
        </div>
        <br>
        <div class="columns text-align-center padding-5">
            <div class="column col-12" v-show="selected_sample && selected_dataset.color_mode">
                <div v-show="selected_rgb && selected_rgb.length>=1" ref="r_slider"></div>
                <div v-show="selected_rgb && selected_rgb.length>=2" ref="g_slider"></div>
                <div v-show="selected_rgb && selected_rgb.length>=3" ref="b_slider"></div>
            </div>
            <div class="column col-12" v-show="selected_sample && !selected_dataset.color_mode && !iOS">
                <div ref="gray_slider"></div>
            </div>
        </div>
        <div id="map"  class="image-view"></div>
        <div class="modal" id="annotation_dialog">
            <a href="#close" class="modal-overlay" aria-label="Close"></a>
            <div class="modal-container">
                <div class="modal-header">
                    <a href="#close" class="btn btn-clear float-right" aria-label="Close"
                        @click="closeDialog('annotation_dialog')"></a>
                    <div class="modal-title h5">Annotation</div>
                </div>
                <div style="padding: .8rem;">
                    <div class="content" style="min-height:100px;">
                        <div class="container margin-bottom-15">
                            <div class="columns">
                                <div class="column col-3 col-sm-6 padding-left-0">
                                    <input class="form-input" type="text"
                                        placeholder="annotation name"
                                        v-model="new_annotation_type.label">
                                </div>
                                <div class="column col-3 col-sm-6 padding-right-0 text-align-right">
                                    <div class="dropdown">
                                        <a href="#" class="btn btn-link dropdown-toggle">
                                            Color: <label
                                                :style="{backgroundColor: new_annotation_type.color}">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
                                            <i class="icon icon-caret"></i>
                                        </a>
                                        <ul class="menu" style="max-height: 200px;">
                                            <li v-for="c in COLORS" :style="{backgroundColor: c}" class="cursor-pointer text-align-left"
                                                @click="new_annotation_type.color = c">{{c}}</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="column col-3 col-sm-6  padding-left-0">
                                    <div class="dropdown">
                                        <a href="#" class="btn btn-link dropdown-toggle">
                                            Type: {{new_annotation_type.type}} <i class="icon icon-caret"></i>
                                        </a>
                                        <ul class="menu" style="max-height: 200px;">
                                            <li v-for="v in ['Polygon', 'LineString', 'Circle', 'Point', 'Label']" class="menu-item"
                                                @click="new_annotation_type.type = v;">
                                                <a href="#">{{v}}</a>
                                            </li>
                                        </ul>
                                    </div>
                                    <div v-if="new_annotation_type.type == 'Label'">
                                        <br>

                                        <div class="has-icon-left">
                                            <input type="text" class="form-input" v-model="new_label" @keyup.enter="addNewLabel()" placeholder="type a new label">
                                            <i class="form-icon icon icon-add"></i>
                                            <span class="chip" v-for="label in new_annotation_type.labels" :key="label">
                                            {{label}}
                                            <a href="#" class="btn btn-clear" aria-label="Close" role="button" @click="removeLabel(label)"></a>
                                        </span>
                                        </div>
                                    </div>
                                </div>
                                <div class="column col-3 col-sm-6 padding-right-0 text-align-right">
                                    <button class="btn btn-primary" @click="addAnnotation(true)">&nbsp;&nbsp;&nbsp;&nbsp; Add
                                &nbsp;&nbsp;&nbsp;&nbsp;</button>
                                </div>
                            </div>
                        </div>
                        <span class="chip" v-for="(anno, k) in annotation_types" :key="k" :style="{backgroundColor: anno.color, cursor: 'pointer',margin: '0px 12px 10px 0'}">
                            {{anno.label}}
                            <a href="#" class="btn btn-clear" aria-label="Close" role="button" @click="removeAnnotation(k)"></a>
                        </span>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn"
                        @click="closeDialog('annotation_dialog')">&nbsp;&nbsp;&nbsp;&nbsp;Close&nbsp;&nbsp;&nbsp;&nbsp;</button>
                </div>
            </div>
        </div>
        <div class="modal" id="modal_file_check">
            <a href="#close" class="modal-overlay" aria-label="Close"></a>
            <div class="modal-container">
                <div class="modal-header">
                    <a href="#close" class="btn btn-clear float-right" aria-label="Close"
                        @click="closeDialog('modal_file_check')"></a>
                    <div class="modal-title h5">Message</div>
                </div>
                <div class="modal-body">
                    <div class="content">
                        {{this.message}}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn" @click="train(true)">OK</button>&nbsp;&nbsp;
                    <button class="btn btn-primary" @click="train(false)">Cancel</button>
                </div>
            </div>
        </div>
        <div class="modal" id="load_samples">
            <a href="#close" class="modal-overlay" aria-label="Close"></a>
            <div class="modal-container">
                <div class="modal-header">
                    <a href="#close" class="btn btn-clear float-right" aria-label="Close"
                        @click="closeDialog('load_samples')"></a>
                    <div class="modal-title h5">Load Samples</div>
                </div>
                <div style="padding: .8rem;">
                    <div class="content" style="min-height:100px;">
                        <div class="columns text-align-center">
                            <div class="column col-6 col-xs-12">
                                <div class="dropdown">
                                    <a href="#" class="btn btn-link dropdown-toggle" tabindex="0">
                                        dataset name:{{selected_dataset_name}} <i
                                            class="icon icon-caret"></i>
                                    </a>
                                    <ul class="menu">
                                        <li v-for="ds in dataset_list" :key="ds" class="menu-item"
                                            @click="datasetSelected(ds)">
                                            <a href="#">
                                                {{ds}}
                                            </a>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                            <div class="column col-6  col-xs-12"><button class="btn" @click="loadSamples(selected_dataset_name); closeDialog('load_samples');">Load
                                    Samples</button></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>
</window>
<style lang="css">
body {
    display: block;
    margin: 2px;
    font-family: Arial, Helvetica, sans-serif;
}
.image-view {
    height: calc(100% - 76px);
}
/* css for the range slider */
@supports (--css: variables) {
    input[type="range"].multirange {
        width: 200%;
    }
    input[type="range"].multirange.ghost {
        --range-color: #5755d9 !important;
    }
}
.cursor-pointer {
    cursor: pointer;
}
.text-align-center {
    text-align: center;
}
.padding-5 {
    padding: 5px;
}
a:hover {
    text-decoration: none !important;
}
.noUi-connect {
    background: #448aff80!important;
}
.noUi-horizontal {
    height: 12px!important;
}
.noUi-target {
    margin-left: 12px!important;
    margin-top: 3px!important;
    margin-bottom: 8px!important;
    margin-right: 12px!important;
}
.noUi-horizontal .noUi-handle {
    height: 20px!important;
    outline: none;
}
.noUi-tooltip {
    display: none!important;
}
.noUi-active .noUi-tooltip {
    display: block!important;
}
.noUi-handle:after, .noUi-handle:before {
    height: 8px!important;
}
.menu .menu-item > a.active, .menu .menu-item > a:active {
    color: rgb(68, 138, 255);
    background: #cde1fd!important;
}
.overlay{
    position: absolute;
    top: 38%;
    left: 50%;
}
.text-align-right {
    text-align: right;
}
.text-align-left {
    text-align: left;
}
.padding-right-0 {
    padding-right: 0;
}
.padding-left-0 {
    padding-left: 0;
}
.margin-bottom-15 {
    margin-bottom: 15px;
}
.icon-size-position {
    font-size: 1.2em;
    position: relative;
    top: 2px;
}
</style>
