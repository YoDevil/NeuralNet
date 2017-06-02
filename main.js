document.addEventListener("DOMContentLoaded", function(event) {

    (function () {
        var old = console.log;
        var logger = document.getElementById('log');
        console.log = function (message) {
            old(message);
            if (typeof message == 'object') {
                logger.innerHTML += (JSON && JSON.stringify ? JSON.stringify(message) : message) + '<br />';
            } else {
                logger.innerHTML += message + '<br />';
            }
        }
    })();

    var pixels_per_square=25;

    // ==================================================================== //
    //                   N E U R A L   N E T   B R A I N                    //
    // ==================================================================== //

    var numInput=48;    //Change according to pixel_per_square: (200/pixel_per_square)*(150/pixel_per_square)
    var numOutput=10;
    var numHidden=Math.floor((numInput+numOutput)/2)

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
        inputs=[];
        outputs=[];
        hOutputs=[];

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
            for(var j=0; j<numOutput; j++){
                var delta = learnRate * oDeltas[j] * hOutputs[i];
                hoWeights[i][j] += delta;
            }
        }

        //  1b) Output biases
        for(var i=0; i<numOutput; i++){
            var delta = learnRate * oDeltas[i] * 1; //  Biases are like neurons of value 1
            oBiases[i] += delta
        }

        //  2a) Input-Hidden weights
        for(var i=0; i<numInput; i++){
            for(var j=0; j<numHidden; j++){
                var delta = learnRate * hDeltas[j] * inputs[i];
                //console.log(learnRate, hDeltas[j], inputs[i], delta);
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
            throw ("Inputs should be as many as targets: " + inputsArray.length + " vs " + targetsArray.length);

        for(var i=0; i<epochs; i++){
            for(var j=0; j<inputsArray.length; j++){
                RunNN(inputsArray[j]);
                BackpropagateNN(targetsArray[j], learnRate);
            }
        }
    }



    // ==================================================================== //
    //                B R A I N   V I S U A L I Z A T I O N                 //
    // ==================================================================== //
	
	var right_container=document.getElementById("right");
	var brain_canvas=document.getElementById("brain-canvas");
	var brain_ctx=brain_canvas.getContext("2d");
	
	//Set canvas to fill parent
	brain_canvas.style.width ='100%';
	brain_canvas.style.height='100%';
	brain_canvas.width  = brain_canvas.offsetWidth;
	brain_canvas.height = brain_canvas.offsetHeight;
	
	brain_ctx.fillStyle="#ffffff";
	brain_ctx.fillRect(0,0,brain_canvas.width,brain_canvas.height);
	
    function DisplayBrain(_inputs, _hidden, _outputs){
        var iPos = [];
        var hPos = [];
        var oPos = [];

        //Reset Canvas
        brain_ctx.fillStyle="#ffffff";
        brain_ctx.fillRect(0,0,brain_canvas.width,brain_canvas.height);

        //Draw 8 input neurons
        var inputsToDraw = 8;
        
        var size = brain_canvas.height/inputsToDraw;
        var spacing = size/4;
        size -= spacing;
        var leftOffset = 10;
        var topOffset = (brain_canvas.height - size*inputsToDraw - spacing*(inputsToDraw-1)) / 2;

        //Draw inputs
        for(var i=0; i<inputsToDraw; i++){
            var centerX = leftOffset + size/2;
            var centerY = topOffset + size/2 + i*(size+spacing);
            DrawCircle(brain_ctx, centerX, centerY, size/2, "#ff0000", "rgba(255,0,0,0.5)");
            var inputText;
            if(_inputs)     inputText = round(_inputs[i], 4);
            else            inputText = "";
            WriteText(brain_ctx, centerX, centerY, inputText, "#000000", size/4);
            iPos[i] = {x: centerX, y: centerY};
        }
        
        //Draw hidden
        spacing+=size;
        for(var i=0; i<inputsToDraw/2; i++){
            var centerX = brain_canvas.width/2;
            var centerY = topOffset + (brain_canvas.height - size * inputsToDraw/2 - spacing * (inputsToDraw/2 - 1))/2 + size/2 + i*(size+spacing);
            DrawCircle(brain_ctx, centerX, centerY, size/2, "#ff0000", "rgba(255,0,0,0.5)");
            var hiddenText;
            if(_hidden)     hiddenText = round(_hidden[i], 4);
            else            hiddenText = "";
            WriteText(brain_ctx, centerX, centerY, hiddenText, "#000000", size/4);
            hPos[i] = {x: centerX, y: centerY};
        }

        //Draw outputs
        spacing+=size;
        for(var i=0; i<inputsToDraw/4; i++){
            var centerX = brain_canvas.width - size/2 - leftOffset;
            var centerY = topOffset + (brain_canvas.height - size * inputsToDraw/4 - spacing * (inputsToDraw/4 - 1))/2 + size/2 + i*(size+spacing);
            DrawCircle(brain_ctx, centerX, centerY, size/2, "#ff0000", "rgba(255,0,0,0.5)");
            var outputText;
            if(_outputs)    outputText = round(_outputs[i], 4);
            else            outputText = "";
            WriteText(brain_ctx, centerX, centerY, outputText, "#000000", size/4);
            oPos[i] = {x: centerX, y: centerY};
        }

        //Draw i-h connections
        for(var j=0; j<inputsToDraw/2; j++){
            for(var i=0; i<inputsToDraw; i++){
                DrawLine(brain_ctx, iPos[i].x, iPos[i].y, hPos[j].x, hPos[j].y, Math.tan(ihWeights[i][j])*10, (Math.sign(ihWeights[i][j])==-1) ? "#00ff00" : "#0066ff", size/2);
            }
        }

        //Draw h-o connections
        for(var j=0; j<inputsToDraw/4; j++){
            for(var i=0; i<inputsToDraw/2; i++){
                DrawLine(brain_ctx, hPos[i].x, hPos[i].y, oPos[j].x, oPos[j].y, Math.tan(hoWeights[i][j])*10, (Math.sign(ihWeights[i][j])==-1) ? "#00ff00" : "#0066ff", size/2);
            }
        }
    }
	
	function DrawCircle (context, centerX, centerY, radius, strokeColor, fillColor){
		context.beginPath();
		context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
		context.fillStyle = fillColor;
		context.fill();
		context.lineWidth = 5;
		context.strokeStyle = strokeColor;
		context.stroke();
	}

    function WriteText (context, centerX, centerY, text, color, size){
        context.fillStyle=color;
        context.textAlign="center";
        context.textBaseline="middle"; 
        context.font=size+"px Arial";
        context.fillText(text, centerX, centerY);
    }

    function DrawLine (context, x1, y1, x2, y2, width, color, radius){
        var angle=Math.atan((x2-x1)/(y2-y1));
        var offsetX=radius*Math.sin(angle);
        var offsetY=radius*Math.cos(angle);
        //offsetX=0;
        //offsetY=0;
        context.strokeStyle=color;
        context.lineWidth=width;
        context.beginPath();
        if(angle>=0){
            context.moveTo(x1+offsetX,y1+offsetY);
            context.lineTo(x2-offsetX,y2-offsetY);
        }
        else{
            context.moveTo(x1-offsetX,y1-offsetY);
            context.lineTo(x2+offsetX,y2+offsetY);
        }
        context.stroke();
    }

    function round(value, decimals) {
        return Number(Math.round(value+'e'+decimals)+'e-'+decimals);
    }


    // ==================================================================== //
    //                N E U R A L N E T   F U N C T I O N S                 //
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
        return 1/(1+Math.exp(-x));
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
    input_ctx.strokeStyle="#000000";
    input_ctx.fillStyle="#ffffff";
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
        var image = document.createElement("img");
        image.setAttribute("src",input_canvas.toDataURL("image/png"));
        image.setAttribute("value",input_number.value);
        image.setAttribute("title",input_number.value);
        image.style.margin=2;
        image.addEventListener("click", function(){
            this.remove();
        });
        input_images.appendChild(image);
        
        input_ctx.clearRect(0,0,input_canvas.width,input_canvas.height);
        input_ctx.fillRect(0,0,input_canvas.width,input_canvas.height);
    }

    // ==================================================================== //
    //              N E U R A L N E T   I N T E G R A T I O N               //
    // ==================================================================== //

    var train_button = document.getElementById("train");
    var run_button = document.getElementById("input-run");
    var run_number = document.getElementById("run-number");
    var guess_button = document.getElementById("input-guess");
    var dataInput = [];
    var dataTarget = [];

    run_button.onclick=function(){
        //Create input values array
        var xValues = getDrawnData();

        //Create output values array
        var tValues = [];
        for(var i=0; i<numOutput; i++){
            tValues.push((run_number.value == i) ? 1 : 0);
        }
        
        var error = MeanSquaredError(xValues, tValues);
        console.log("Error on new data = " + error);

        DisplayBrain(xValues, hOutputs, outputs);
    }

    guess_button.onclick=function(){
        //Create input values array
        var xValues = getDrawnData();

        RunNN(xValues);
        var guessedNumber = getMax(outputs);

        console.log("Hai disegnato un " + guessedNumber[0] + ", Sono sicuro al " + Math.round(guessedNumber[1]*1000)/10 + "%.");

        DisplayBrain(xValues, hOutputs, outputs);
    }

    function getDrawnData(){
        //Create data matrix from canvas
        var matrix = [];
        for(var i = 0; i < input_canvas.width; i++){
            matrix[i] = [];
            for(var j = 0; j < input_canvas.height; j++){
                var imageData = input_ctx.getImageData(i, j, 1, 1);
                var data = imageData.data;
                matrix[i][j] = (data[0] == 0 && data[1] == 0 && data[2] == 0) ? 1 : 0;
            }
        }

        //Create scaled matrix from data matrix
        var scaledMatrix = [];
        for(var i=0; i<input_canvas.width; i++){
            if(!scaledMatrix[parseInt(i/pixels_per_square)]) 
                scaledMatrix[parseInt(i/pixels_per_square)]=[];
            for(var j=0; j<input_canvas.height; j++){
                if(!scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)]) 
                    scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)]=0;
                scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)] += matrix[i][j];
            }
        }
        
        //Create input values array
        var xValues = [];
        for(var i=0; i<parseInt(input_canvas.width/pixels_per_square); i++){
            for(var j=0; j<parseInt(input_canvas.height/pixels_per_square); j++){
                var normalizedInput = 2*scaledMatrix[i][j]/(pixels_per_square*pixels_per_square) - 1;   //Between -1 and +1
                xValues.push(normalizedInput);
            }
        }

        return xValues;
    }

    train_button.onclick=function(){
        var run = function(){
            //console.log(dataInput);
            //console.log(dataTarget);
            var loop = setInterval(function(){
                TrainNN(dataInput, dataTarget, 1, -0.05);
                DisplayBrain(null,null,null);
                if(MeanSquaredError(dataInput[0], dataTarget[0])<0.001){
                    clearInterval(loop);
                    console.log("Training completato<br>Errore quadratico medio = "+MeanSquaredError(dataInput[0], dataTarget[0]));
                }
            },10)
        }
        makeData(run);
    }

    function makeData(callback){
        dataInput = [];
        var images = input_images.childNodes;
        
        var loaders = [];

        for(var z=0; z<images.length; z++){
            //  Create target data
            dataTarget[z]=[];
            for(var i=0; i<numOutput; i++){
                dataTarget[z].push((images[z].getAttribute("value") == i) ? 1 : 0);
            }

            //  Create input data
            loaders.push(createInputData(images[z],z))
        }

        $.when.apply(null, loaders).done(function(){
            callback();
        });
    }

    function createInputData(imageElement, index){
        var deferred = $.Deferred();
        var img = new Image;
        img.onload = function(){
            getDataFromImage(img, index)
            deferred.resolve();
        }
        img.src = imageElement.src;
        return deferred.promise();
    }

    function getDataFromImage(img, index){
        var matrix = matrixFromImage(img, img.width, img.height);
        var scaledMatrix = [];
        for(var i=0; i<img.width; i++){
            if(!scaledMatrix[parseInt(i/pixels_per_square)]) 
                scaledMatrix[parseInt(i/pixels_per_square)]=[];
            for(var j=0; j<img.height; j++){
                if(!scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)]) 
                    scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)]=0;
                scaledMatrix[parseInt(i/pixels_per_square)][parseInt(j/pixels_per_square)] += matrix[i][j];
            }
        }
        dataInput[index]=[];
        for(var i=0; i<parseInt(img.width/pixels_per_square); i++){
            for(var j=0; j<parseInt(img.height/pixels_per_square); j++){
                var normalizedInput = 2*scaledMatrix[i][j]/(pixels_per_square*pixels_per_square) - 1;   //Between -1 and +1
                dataInput[index].push(normalizedInput);
            }
        }
    }


    function matrixFromImage(img, width, height) {
        var matrix = [];
        var canvas = document.createElement("canvas");
        canvas.width=width;
        canvas.height=height;
        var ctx = canvas.getContext("2d");

        ctx.drawImage(img, 0, 0, width, height);

        for(var i = 0; i < width; i++){
            matrix[i] = [];
            for(var j = 0; j < height; j++){
                var imageData = ctx.getImageData(i, j, 1, 1);
                var data = imageData.data;
                matrix[i][j] = (data[0] == 0 && data[1] == 0 && data[2] == 0) ? 1 : 0;
            }
        }
        return matrix;
    }


    function getMax(array)
    {
        var maxIndex = 0;
        var maxVal = array[0];
        for (var i = 0; i < array.length; i++)
        {
            if (array[i] > maxVal)
            {
                maxVal = array[i]; maxIndex = i;
            }
        }
        return [maxIndex,maxVal];
    }
});