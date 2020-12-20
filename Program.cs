using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MNIST_S
{
    class Program
    {
        private static String setFileName()
        {
            String FileDirName = "C:\\Users\\AFOC\\Documents\\Visual Studio 2008\\Projects\\MNIST_S\\MNIST_S\\TrainData\\";
            
            Random r = new Random();
            int cuN = r.Next(10);
            int cuI = r.Next(1280);
            String FileName = cuN.ToString() + "-" + cuI.ToString() + ".txt";
            return FileDirName + FileName;
        }

        private static void ShowAsPixel(double[] SingleData, StreamWriter sw)
        {
            for (int i = 0; i < 28; i++)
            {
                for (int j = i * 28; j < i * 28 + 28; j++)
                {
                    sw.Write(SingleData[j]);
                }
                sw.WriteLine();
            }
            sw.Write(SingleData[784]);
            sw.WriteLine();
        }

        private static void getPixelData(double[] SingleData, StreamReader reader)
        {
            for (int i = 0; i < 785; i++)
            {
                SingleData[i] = reader.Read() - '0';
                reader.Read();
            }
        }

        static void Main(string[] args)
        {
            StreamReader TestDataReader;
            StreamWriter ResultWriter = new StreamWriter("C:\\Users\\AFOC\\Documents\\Visual Studio 2008\\Projects\\MNIST_S\\MNIST_S\\result.txt");
            double[] SingleData = new double[785];
            int data_amount = 100000;
            MNIST_Train trainer = new MNIST_Train(ResultWriter, data_amount);

            for (int i = 0; i < data_amount; i++)
            {
                if (i >= 50000) { trainer.flag = 1; }
                ResultWriter.WriteLine("[Train Set : "+(i+1).ToString()+"]");
                TestDataReader = new StreamReader(setFileName());
                getPixelData(SingleData, TestDataReader);

                // Console.WriteLine();
                // Console.WriteLine("================================");
                // ShowAsPixel(SingleData);
                // Console.WriteLine("================================");
                // Console.WriteLine();
                
                
                System.Threading.Thread.Sleep(1);

                // Initialize model
                trainer.setInput(SingleData);
                ShowAsPixel(SingleData, ResultWriter);
                trainer.MNIST_TrainActivity();
                if (i % 100 == 0)
                {
                    Console.WriteLine("train idx : " + (i + 1));
                    Console.WriteLine("Loss function : " + trainer.objValue);
                    Console.WriteLine();
                }
                trainer.clearValue();
            }
            double accurate = (double)trainer.getAnsCnt() / data_amount * 100;
            ResultWriter.WriteLine("Train end");
            ResultWriter.WriteLine("Accurate : "+accurate+"%");
            ResultWriter.Close();
        }
    }
}
