using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.IO;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using System.Windows;
using DeepLearningDraft.NN;

namespace DeepLearningDraft
{
    /// <summary>
    /// App.xaml の相互作用ロジック
    /// </summary>
    public partial class App : Application
    {
        public static readonly Random rand = new Random();

        protected override void OnStartup(StartupEventArgs e)
        {
            // foobar()

            /*
            var nn = new NN.NN(
                new IntFuncPair(28 * 28, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(10, ActivationFunction.Sigmoid)
                );
            nn.Dump();

            var dataset = new ImageDataset(@"C:\Users\Kent2\Desktop\MyProgram\WPF\DeepLearningDraft\archive\");

            Log.Line("Test: ");
            for (int i = 0; i < 10; i++)
                nn.Calculate(new NN.Matrix(dataset.GetSample(i, false).inputs)).Dump();
            */

            Matrix a = new Matrix(new double[,] { { 2, 5 }, { -1, 0 } });
            Matrix b = new Matrix(new double[,] { { -4, 1 }, { 3, 7 } });
            Log.Line("a + b =");
            (a + b).Dump();

            return;
            
            var simpleDataset = new HalfAdderDataset();
            var nn = new NN.NN(
                8,
                new IntFuncPair(2, ActivationFunction.Sigmoid),
                new IntFuncPair(3, ActivationFunction.Sigmoid),
                new IntFuncPair(2, ActivationFunction.Sigmoid)
                );
            
            /*
            var simpleDataset = new ImageDataset(@"C:\Users\Kent2\Desktop\MyProgram\WPF\DeepLearningDraft\archive\");
            var nn = new NN.NN(
                1d,
                new IntFuncPair(28 * 28, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(10, ActivationFunction.Sigmoid)
                );*/
            Log.Line("Simple test");

            for (int i = 0; i < 4; i++)
                nn.Calculate(new NN.Matrix(simpleDataset.GetSample(i, false).inputs)).Dump();

            Log.Line("Dump weights and biases");
            nn.Dump();

            while(false)
            for(int batch = 0; batch < 1; batch++)
            {
                int batchNum = 4;
                var inputs = new NN.Matrix[batchNum];
                var answers = new NN.Matrix[batchNum];

                for(int i = 0; i < batchNum; i++)
                {
                    var datasetIndex = batch * batchNum + i;
                    var pair = simpleDataset.GetSample(datasetIndex, false);
                    inputs[i] = pair.inputs.Clone();
                    answers[i] = pair.desiredOutputs.Clone();
                }

                nn.GradientDecent(inputs, answers, 0.21);
                nn.EvaluateByLoss(inputs, answers);
                    nn.EvaluateByQuestions(inputs, answers, (output, answer) =>
                        {
                            var (oR, oC, oD) = output.MaxCell();
                            var (aR, aC, aD) = answer.MaxCell();
                            return oR == aR;
                        });
            }
        }
    }
}
