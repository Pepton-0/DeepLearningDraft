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

            var simpleDataset = new HalfAdderDataset();
            var nn = new NN.NN(
                1,
                new IntFuncPair(2, ActivationFunction.Sigmoid),
                new IntFuncPair(3, ActivationFunction.Sigmoid),
                new IntFuncPair(2, ActivationFunction.Sigmoid)
                );
            Log.Line("Simple test");

            for (int i = 0; i < 4; i++)
                nn.Calculate(new NN.Matrix(simpleDataset.GetSample(i, false).inputs)).Dump();

            while(true)
            for(int batch = 0; batch < 1; batch++)
            {
                int batchNum = 4;
                var inputs = new NN.Matrix[batchNum];
                var answers = new NN.Matrix[batchNum];

                for(int i = 0; i < batchNum; i++)
                {
                    var datasetIndex = batch * batchNum + i;
                    var pair = simpleDataset.GetSample(datasetIndex, false);
                    inputs[i] = new NN.Matrix(pair.inputs);
                    answers[i] = new NN.Matrix(pair.desiredOutputs);
                }

                Log.Line($"Current Loss: {nn.LossAvgFromInputs(inputs, answers)}");
                nn.GradientDecent(inputs, answers, 1);
                nn.EvaluateByLoss(inputs, answers);
            }
        }
    }
}
