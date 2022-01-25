using NeuralNetLibrary;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetWinForms
{
    public partial class Form1 : Form
    {
        int trainCount = 0;
        public Form1()
        {
            InitializeComponent();
        }

        double TestingFunction(double x)
        {
            //return Math.Cos(x * Math.PI / 2.0);
            //return (Math.Sin(x * Math.PI / 1.0)+1.0)/2.0;// l=10
            return (Math.Sin(x * Math.PI * 2) + 1.0) / 2.0; //l=20
            //return x * x+x;
        }

        internal NeuralNet CreateAndTrainNet()
        {
            Random random = new Random();

            NeuralNet neuralNet = new(random);
            neuralNet.AddLayer(1);
            neuralNet.AddLayer(20, ActivationTypes.Sigmoid);
            neuralNet.AddLayer(20, ActivationTypes.Sigmoid);
            neuralNet.AddLayer(1, ActivationTypes.Sigmoid);

            Queue<double> errorsQueue = new();
            for (trainCount = 0; trainCount < 1_000_000; trainCount++)
            {
                double d = random.NextDouble();
                var input = Vector<double>.Build.DenseOfArray(new double[] { d });
                var outputToBe = Vector<double>.Build.DenseOfArray(new double[] { TestingFunction(d) });

                neuralNet.Train(input, outputToBe);

                double errorMSE = neuralNet.ErrorMSE(outputToBe);
                errorsQueue.Enqueue(errorMSE);
                if (errorsQueue.Count > 10 && errorsQueue.Average() < 0.001) break;
                if (errorsQueue.Count > 50) errorsQueue.Dequeue();
            }
            return neuralNet;
        }



        void DrawImage()
        {
            NeuralNet neuralNet = CreateAndTrainNet();
            neuralNet.ClearAllButWeight();
            NeuralNet.SaveJson(@"D:\Temp\net.json", neuralNet);

            int w = sharpPictureBox1.Width;
            int h = sharpPictureBox1.Height;
            double y;
            Bitmap bitmap = new Bitmap(w, h);

            int minSize = Math.Min(w, h) - 1;
            int j;

            Vector<double> input = Vector<double>.Build.Dense(1);
            for (int i = 1; i < minSize; i++)
            {
                //bitmap.SetPixel(i, h - 1 - i, Color.DarkGray);

                double x = (double)i / (double)minSize;
                y = TestingFunction(x);
                j = (int)(y * (double)minSize);
                if (j > 0 && j < minSize) bitmap.SetPixel(i, h - 1 - j, Color.Black);


                input[0] = x;
                y = neuralNet.Forward(input)[0];
                j = (int)(y * (double)minSize);
                if (j > 0 && j < minSize) bitmap.SetPixel(i, h - 1 - j, Color.Red);
            }

            sharpPictureBox1.Image = bitmap;
            sharpPictureBox1.Refresh();
            Text = trainCount.ToString();
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            timer1.Stop();
            DrawImage();
        }
    }
}