using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MNIST_S
{
    class Layer
    {
        public double[] inputValue;
        int iHeight; // iWidth == 1

        public double[,] weight;
        int wHeight, wWidth; // wHeight = oHeight, wWidth = iHeight

        public double[] outputValue;
        public int oHeight; // oWidth == 1

        public double[] bias;
        public double[] zValue;
        public double[] deltaValue;
        public double[,] weightPropaValue;
        public double[] biasPropaValue;

        public Layer(int iHeight, int oHeight)
        {
            this.iHeight = iHeight;
            this.oHeight = oHeight;
            wHeight = oHeight;
            wWidth = iHeight;
            inputValue = new double[iHeight];
            weight = new double[wHeight, wWidth];
            outputValue = new double[oHeight];
            bias = new double[oHeight];
            zValue = new double[oHeight];
            weightPropaValue = new double[wHeight, wWidth];
            biasPropaValue = new double[oHeight];
            deltaValue = new double[oHeight];
            setLayerEntity();
        }

        public void layerActivity()
        {
            calcMetrix();
            sigmoidActivity();
        }

        public void inputActivity()
        {
            outputValue = inputValue;
        }

        public double sigmoid(double x)
        {
            return 1 / (1 + (double)Math.Exp(-x));
        }

        public double sigmoidDerivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        public void sigmoidActivity()
        {
            for (int i = 0; i < oHeight; i++)
            {
                outputValue[i] = sigmoid(outputValue[i]);
            }
        }

        public void setInputValue(double[] input){ inputValue= input; }
        public void setLayerEntity()
        {
            Random r = new Random();
            int pm = 1;
            for (int i = 0; i < oHeight; i++)
            {
                if (r.NextDouble() > 0.5) { pm *= -1; }
                bias[i] = r.NextDouble()*pm;
                for (int j = 0; j < iHeight; j++)
                {
                    if (r.NextDouble() > 0.5) { pm *= -1; }
                    weight[i, j] = r.NextDouble()*pm;
                }
            }
        }

        public void calcMetrix()
        {
            // iWidth by iHegiht * wWidth by wHeight
            int i, j;
            for (i = 0; i < wHeight; i++)
            {
                for (j = 0; j < iHeight; j++)
                {
                    outputValue[i] += inputValue[j] * weight[i, j];
                }
                outputValue[i] += bias[i];
                zValue[i] = outputValue[i];
            }
        }

        public double softmax()
        {
            double expSum = 0.0;
            for (int i = 0; i < outputValue.Length; i++)
            {
                expSum += Math.Log(outputValue[i]);
            }
            for (int i = 0; i < outputValue.Length; i++)
            {
                outputValue[i] = Math.Log(outputValue[i]) / expSum;
            }
            return 0;
        }

        public void ReLU()
        {
            int i;
            for (i = 0; i < 10; i++)
            {
                if (outputValue[i] < 0) { outputValue[i] = 0; }
            }
        }
    }
}
