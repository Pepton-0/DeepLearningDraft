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
                nn.GradientDecent(inputs, answers, 0.3);
                nn.EvaluateByLoss(inputs, answers);
            }
        }

        private void foobar()
        {
            double Foo(double a)
            {
                return a * a * 3;
            }

            Log.Line("Diff: " + NMath.Differential(d => Foo(d), 4));

            var network1 = new NeuralNetwork(
                new IntFuncPair(28 * 28, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(8, ActivationFunction.Sigmoid),
                new IntFuncPair(10, ActivationFunction.Sigmoid)
                );

            void Test3(Matrix inputs, Matrix desiredOutputs)
            {
                Matrix result = network1.Calculate(inputs);
                double cost = network1.CalculateCostFromOutputs(result, desiredOutputs);
                string str = "";
                for (int i = 0; i < inputs.Rows; i++)
                {
                    str += $"{inputs[i, 0]}";
                    if (i != inputs.Rows - 1)
                    {
                        str += ", ";
                    }
                }
                string resultStr = "  -> ";
                for (int i = 0; i < result.Rows; i++)
                {
                    resultStr += $"{result[i, 0]}";
                    if (i != result.Rows - 1)
                    {
                        resultStr += ", ";
                    }
                }
                Log.Line(str);
                Log.Line(resultStr);
                Log.Line($" -> Cost: {cost}");
                Log.Line("");
            }

            var dataset = new ImageDataset(@"C:\Users\Kent2\Desktop\MyProgram\WPF\DeepLearningDraft\archive\");
            for (int i = 0; i < 5; i++)
            {
                var v = dataset.GetSample(i, false);
                Test3(v.inputs, v.desiredOutputs);
            }

            double prevCost = 0d;
            double nextCost = double.PositiveInfinity;
            for (int f = 0; f < 15; f++)
            {
                int batch = 8;
                Matrix[] inputs = new Matrix[batch];
                Matrix[] answers = new Matrix[batch];
                for (int i = 0; i < batch; i++)
                {
                    var sample = dataset.GetSample(f * batch + i, false);
                    inputs[i] = sample.inputs;
                    answers[i] = sample.desiredOutputs;
                }

                do
                {
                    prevCost = nextCost;
                    nextCost = network1.CalculateGradient(inputs, answers);
                    Log.Line($"Cost: {nextCost} [{nextCost - prevCost}] with Gradient");
                } while (!NMath.Approximately(prevCost, nextCost));


                int counter = 0;
                do
                {
                    counter++;
                    if (counter > 1)
                        break;
                    prevCost = nextCost;
                    nextCost = network1.Backpropagate(inputs, answers);
                    Log.Line($"Cost: {nextCost} [{nextCost - prevCost}] with Backpropagate");
                } while (!NMath.Approximately(prevCost, nextCost));

                Log.Line("Cost is " + nextCost);
            }

            /*
            Log.Line("----- Start Backpropagation -----");
            for (int f = 0; f < 3; f++)
            {
                int batch = 8;
                double prevCost = 0d;
                double nextCost = double.PositiveInfinity;
                Matrix[] inputs = new Matrix[batch];
                Matrix[] answers = new Matrix[batch];
                for (int i = 0; i < batch; i++)
                {
                    var sample = dataset.GetSample(f * batch + i, false);
                    inputs[i] = sample.inputs;
                    answers[i] = sample.desiredOutputs;
                }

                do
                {
                    prevCost = nextCost;
                    nextCost = network1.Backpropagate(inputs, answers);
                    Log.Line("WIP Cost is " + nextCost);
                } while (!Approximately(prevCost, nextCost));

                Log.Line("Cost is " + nextCost);
            }*/

            int ans = 0;
            for (int i = 0; i < dataset.GetSampleCount(true); i++)
            {
                var predict = network1.Calculate(dataset.GetSample(i, true).inputs);
                double highest = double.NegativeInfinity;
                int j = 0;
                int ind = 0;


                double highest1 = double.NegativeInfinity;
                int j1 = 0;
                int ind1 = 0;


                predict.FilterFunc(from =>
                {
                    if (highest < from)
                    {
                        highest = from;
                        ind = j;
                    }
                    j++;
                    return from;
                });
                dataset.GetSample(i, true).desiredOutputs.FilterFunc(from =>
                {
                    if (highest1 < from)
                    {
                        highest1 = from;
                        ind1 = j1;
                    }
                    j1++;
                    return from;
                });
                if (ind == ind1)
                {
                    ans++;
                }
            }
            Log.Line($"Accuracy: {((double)ans / (double)dataset.GetSampleCount(true)) * 100d}%");
        }
    }
}
