using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MNIST_S
{
    class MNIST_Train
    {
        public int flag = 0;
        double[] input = new double[784];
        double[] output;
        public int[] OneHot = new int[10];
        public double trainRate = 0.1;
        double entropy;
        public double objValue;
        int data_amount;
        int ans_cnt = 0;
        StreamWriter sw;
        Layer inputLayer = new Layer(784, 512);
        Layer hiddenLayer = new Layer(512, 10);
        Layer outputLayer = new Layer(10, 10);

        public MNIST_Train(StreamWriter sw, int data_amount)
        {
            this.sw = sw;
            this.data_amount = data_amount;
        }

        public int getAnsCnt()
        {
            return ans_cnt;
        }

        public void MNIST_TrainActivity()
        {
            // input layer
            inputLayer.setInputValue(input);
            inputLayer.inputActivity();

            // hidden layer
            hiddenLayer.setInputValue(inputLayer.outputValue);
            hiddenLayer.layerActivity();

            // output layer
            outputLayer.setInputValue(hiddenLayer.outputValue);
            outputLayer.layerActivity();
            outputLayer.softmax();
            output = outputLayer.outputValue;

            // calculate entropy
            entropy = crossEntropy(outputLayer.outputValue, OneHot);
            // print result
            printOutput(outputLayer.outputValue);
            sw.WriteLine("Entropy : " + entropy.ToString());
            // backpropagation
            objValue = ObjFunction();
            backpropagation();
        }

        public void backpropagation()
        {
            int i, j, m;
            // output->hidden
            for (i = 0; i < outputLayer.oHeight; i++)
            {
                // delta^L_i
                outputLayer.deltaValue[i] = (output[i] - OneHot[i]) * outputLayer.sigmoidDerivative(outputLayer.zValue[i]);
                outputLayer.biasPropaValue[i] = outputLayer.deltaValue[i] * trainRate;
                // delta_i * a_j
                for (j = 0; j < hiddenLayer.oHeight; j++)
                {
                    outputLayer.weightPropaValue[i, j] = outputLayer.deltaValue[i] * hiddenLayer.outputValue[j];
                    outputLayer.weightPropaValue[i, j] *= trainRate;
                }
            }

            for (i = 0; i < outputLayer.oHeight; i++)
            {
                for (j = 0; j < hiddenLayer.oHeight; j++)
                {
                    outputLayer.weight[i, j] += outputLayer.weightPropaValue[i, j];
                }
            }

            // hidden->hidden
            for(i=0; i<hiddenLayer.oHeight; i++){
                hiddenLayer.deltaValue[i] = 0;
                for(m=0; m<outputLayer.oHeight; m++){
                    hiddenLayer.deltaValue[i] += outputLayer.deltaValue[m]*outputLayer.weight[i,m];
                }
                hiddenLayer.deltaValue[i] *= hiddenLayer.sigmoidDerivative(hiddenLayer.zValue[i]);
                hiddenLayer.biasPropaValue[i] = hiddenLayer.deltaValue[i] * trainRate;
            }

            // hidden->input
            for (i = 0; i < hiddenLayer.oHeight; i++)
            {
                for (j = 0; j < inputLayer.oHeight; j++)
                {
                    hiddenLayer.weightPropaValue[i, j] = hiddenLayer.deltaValue[i] * inputLayer.outputValue[j];
                    hiddenLayer.weightPropaValue[i, j] *= trainRate;
                }
            }
            
            for (i = 0; i < hiddenLayer.oHeight; i++)
            {
                for (j = 0; j < inputLayer.oHeight; j++)
                {
                    hiddenLayer.weight[i, j] += hiddenLayer.weightPropaValue[i, j];
                }
            }
        }

        public void clearValue()
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = 0;
            }
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0;
            }
            for (int i = 0; i < OneHot.Length; i++)
            {
                OneHot[i] = 0;
            }
            entropy = 0;
            objValue = 0;
        }

        public void setInput(double[] input)
        {
            this.input = input;
            int i = 0;
            for (i = 0; i < 10; i++)
            {
                if (i == input[784])
                {
                    OneHot[i] = 1;
                }
            }
        }
        
        public double crossEntropy(double[] output, int[] OneHot)
        {
            double cEntropy = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cEntropy += OneHot[i] * Math.Log(output[i], Math.E) * (-1);
                cEntropy += (1 - OneHot[i]) * Math.Log(1 - output[i], Math.E) * (-1);
            }
            cEntropy /= data_amount;
            return cEntropy;
        }

        public StreamWriter getSW()
        {
            return sw;
        }

        public double ObjFunction()
        {
            double c = 0;
            for (int i = 0; i < 10; i++)
            {
                c += Math.Pow((output[i] - OneHot[i]), 2);
            }
            sw.WriteLine("Loss Function : " + c);
            sw.WriteLine();
            return c;
        }


        public void printOutput(double[] outputMetrix)
        {
            int i;
            sw.WriteLine("Answer : " + input[784].ToString() + " ");
            /*for (i = 0; i < 10; i++)
            {
                sw.WriteLine(i.ToString() + " : " + outputMetrix[i].ToString());
            }*/
            showResult(outputMetrix, input[784]);
        }



        public void showResult(double[] output, double answer)
        {
            double max = 0.0;
            int maxIndex = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > max)
                {
                    max = output[i];
                    maxIndex = i;
                }
            }
            sw.WriteLine("My Answer : " + maxIndex.ToString());
            if (answer.ToString() == maxIndex.ToString() && flag == 1) { ans_cnt++; }
        }
    }
}
