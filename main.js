document.addEventListener("DOMContentLoaded", function(event) {

    // ==================================================================== //
    //                   N E U R A L   N E T   B R A I N                    //
    // ==================================================================== //

    var numInput=3;
    var numHidden=5;
    var numOutput=2;

    var inputs = createArray(numInput);
    var ihWeights = createArray(numInput,numHidden); //matrix
    var hoWeights = createArray(numHidden,numOutput); //matrix
    var hBiases = createArray(numHidden);
    var hOutputs = createArray(numHidden);
    var oBiases = createArray(numOutput);
    var outputs = createArray(numOutput);
    var oDeltas = createArray(numOutput);
    var hDeltas = createArray(numHidden);

    //  Initialize neural net
    (function (){
        //  1) Set random weights
        var max=1/Math.sqrt(numInput);
        var min=-max;
        for(var i=0; i<numInput; i++){
            for(var j=0; j<numHidden; j++){
                ihWeights[i][j]=Math.random()*(max-min) + min;
            }
        }
        for(var i=0; i<numHidden; i++){
            for(var j=0; j<numOutput; j++){
                hoWeights[i][j]=Math.random()*(max-min) + min;
            }
        }

        // 2) Set random biases
        for(var i=0; i<numHidden; i++){
            hBiases[i]=Math.random()*(max-min) + min;
        }
        for(var i=0; i<numOutput; i++){
            oBiases[i]=Math.random()*(max-min) + min;
        }
    })()

    /**
     * Calculates the outputs and populate the 'outputs' variable.
     * @param {Number[]} xValues
     * @returns {Number[]} outputs
     */
    function RunNN(xValues){
        if(xValues.length != numInput)
            throw "Input values array has a different size than inputs array";

        //  Copy inputs to global variable
        for(var i=0; i< xValues.length; i++){
            inputs[i] = xValues[i];
        }
        
        //  Calculate the outputs for hidden neurons
        for(var j=0; j<numHidden; j++){
            var sum = 0;
            for(var i=0; i<numInput; i++){
                sum+=inputs[i]*ihWeights[i][j];
            }
            sum+=hBiases[j];
            hOutputs[j]=sigmoid(sum);
        }

        //  Calculate the outputs for the output neurons
        for(var j=0; j<numOutput; j++){
            var sum = 0;
            for(var i=0; i<numHidden; i++){
                sum+=hOutputs[i]*hoWeights[i][j];
            }
            sum+=oBiases[j];
            outputs[j]=sigmoid(sum);
        }
        
        return outputs;
    }

    /**
     * 
     * @param {Number[]} tValues 
     * @param {Number} learnRate 
     */
    function BackpropagateNN(tValues, learnRate){
        if(tValues.length != numOutput)
            throw "Target values array has a different size than outputs array";
        
        //  Calculate deltas
        //  1) Output deltas
        for(var i=0; i<numOutput; i++){
            // Derivative of the Error function
            oDeltas[i] = outputs[i] * (1-outputs[i]) * -(tValues[i]-outputs[i]);
        }

        //  2) Hidden deltas
        for(var i=0; i<numHidden; i++){
            var sum = 0;
            for(var j=0; j<numOutput; j++){
                sum+=oDeltas[j]*hoWeights[i][j];
            }
            hDeltas[i] = hOutputs[i] * (1-hOutputs[i]) * sum;
        }

        //  Update weights
        //  1a) Hidden-Output weights
        for(var i=0; i<numHidden; i++){
            for(var j=0; i<numOutput; i++){
                var delta = learnRate * oDeltas[j] * hOutputs[i];
                hoWeights[i][j] += delta;
            }
        }

        //  1b) Output biases
        for(var i=0; i<numOutput; i++){
            var delta = learnRate * oDeltas[j] * 1; //  Biases are like neurons of value 1
            oBiases[i] += delta
        }

        //  2a) Input-Hidden weights
        for(var i=0; i<numInput; i++){
            for(var j=0; j<numHidden; j++){
                var delta = learnRate * hDeltas[j] * inputs[i];
                ihWeights[i][j] += delta;
            }
        }

        //  2b) Hidden biases
        for(var i=0; i<numHidden; i++){
            var delta = learnRate * hDeltas[i] * 1; //  Biases are like neurons of value 1
            hBiases[i] += delta;
        }
    }

    /**
     * 
     * @param {Number[]} xValues 
     * @param {Number[]} tValues 
     * @returns {Number}
     */
    function MeanSquaredError(xValues, tValues){
        if(tValues.length != numOutput)
            throw "Target values array has a different size than outputs array";

        //  Calculate Outputs
        RunNN(xValues);

        //  Calculate Error
        var sum = 0;
        for(var i=0; i<numOutput; i++){
            sum += Math.pow(tValues[i] - outputs[i],2)
        }
        var error = sum/2;  //Error function is 1/2 sumof(Tk-Ok)^2

        return error;
    }

    /**
     * 
     * @param {Number[][]} inputsArray 
     * @param {Number[][]} targetsArray 
     * @param {Number} epochs 
     * @param {Number} learnRate 
     */
    function TrainNN(inputsArray, targetsArray, epochs, learnRate){
        if(inputsArray.length!=targetsArray.length)
            throw "Inputs should be as many as targets";

        for(var i=0; i<epochs; i++){
            for(var j=0; j<inputsArray.length; j++){
                RunNN(inputsArray[j]);
                BackpropagateNN(targetsArray[i]);
            }
        }
    }



    // ==================================================================== //
    //                          F U N C T I O N S                           //
    // ==================================================================== //

    function createArray(length) {
        var arr = new Array(length || 0),
            i = length;
        if (arguments.length > 1) {
            var args = Array.prototype.slice.call(arguments, 1);
            while(i--) arr[length-1 - i] = createArray.apply(this, args);
        }
        return arr;
    }

    function sigmoid(x){
        return 1/1+Math.exp(-x);
    }




    // ==================================================================== //
    //                            P R O G R A M                             //
    // ==================================================================== //

    var input_canvas = document.getElementById("input");
    var input_ctx = input_canvas.getContext("2d");
    var input_clear = document.getElementById("input-clear");
    var input_number = document.getElementById("input-number");
    var input_add = document.getElementById("input-add");
    var input_images = document.getElementById("input-images");
    var input_points = [];
    var isDrawing = false;

    input_ctx.lineWidth=10;
    input_ctx.lineJoin = "round";
    input_ctx.strokeStyle="#000";
    input_ctx.fillStyle="#fff";
    input_ctx.fillRect(0,0,input_canvas.width,input_canvas.height);

    input_canvas.onmousedown=function(e){
        input_points = [];
        isDrawing = true;
        var mouseX = e.pageX-this.offsetLeft;
        var mouseY = e.pageY-this.offsetTop;
        input_points.push({x:mouseX,y:mouseY});
    };

    input_canvas.onmousemove=function(e){
        if(!isDrawing) return;
        var mouseX = e.pageX-this.offsetLeft;
        var mouseY = e.pageY-this.offsetTop;
        input_points.push({x:mouseX,y:mouseY});
        updateDrawing();
    }

    input_canvas.onmouseup=function(e){
        if(!isDrawing) return;
        var mouseX = e.pageX-this.offsetLeft;
        var mouseY = e.pageY-this.offsetTop;
        input_points.push({x:mouseX,y:mouseY});
        updateDrawing();
        isDrawing=false;
    }
    input_canvas.onmouseleave=input_canvas.onmouseup;

    function updateDrawing(){
        input_ctx.beginPath();
        input_ctx.moveTo(input_points[0].x,input_points[0].y);
        for(var i=1; i<input_points.length;i++){
            input_ctx.lineTo(input_points[i].x,input_points[i].y);
        }
        input_ctx.stroke();
    }

    input_clear.onclick=function(){
        input_ctx.clearRect(0,0,input_canvas.width,input_canvas.height);
        input_ctx.fillRect(0,0,input_canvas.width,input_canvas.height);
    }

    input_add.onclick=function(){
        console.log("add");
        var image = document.createElement("img");
        image.setAttribute("src",input_canvas.toDataURL("image/png"));
        image.setAttribute("value",input_number.value);
        image.addEventListener("click", function(){
            this.remove();
        });
        input_images.appendChild(image);
        
        input_ctx.clearRect(0,0,input_canvas.width,input_canvas.height);
        input_ctx.fillRect(0,0,input_canvas.width,input_canvas.height);
    }
});