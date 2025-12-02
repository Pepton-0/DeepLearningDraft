using CommunityToolkit.Mvvm.DependencyInjection;
using DeepLearningDraft.Models;
using DeepLearningDraft.Models.Services;
using DeepLearningDraft.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.IO;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

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
            ImageLearn();
            // MathLearnAdam();

            // Prepare MVVM application
            Ioc.Default.ConfigureServices(new ServiceCollection()
                .AddSingleton<IConfigService, ConfigService>()
                .AddSingleton<IConductor, Conductor>()
                .AddTransient<MainWindowViewModel>()
                .BuildServiceProvider());

            new MainWindow().Show();
        }

        static void ImageCheck()
        {
            var dataset = new ImageDataset("C:\\Users\\Kent2\\Desktop\\MyProgram\\WPF\\DeepLearningDraft\\archive");
            for (int i = 0; i < 10; i++)
            {
                (var input, var desiredOutput) = dataset.GetSample(i, false);
                Log.Line($"Sample{i}");
                dataset.DumpImage(i, false);
                Log.Line($"The number is : {desiredOutput.MaxCell().r}");
            }
        }

        static void ImageLearn()
        {
            var dataset = new ImageDataset("C:\\Users\\Kent2\\Desktop\\MyProgram\\WPF\\DeepLearningDraft\\archive");
            (var input, var desiredOutput) = dataset.GetSample(0, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample answer:");
            desiredOutput.Dump();

            var nn = NN.CreateFromFileOrNew("nn.xml", 8,
                LossFunction.CrossEntropy,
                new IntFuncPair(28 * 28, ActivationFunction.ReLu),
                new IntFuncPair(512, ActivationFunction.ReLu),
                new IntFuncPair(128, ActivationFunction.Linear),
                new IntFuncPair(10, ActivationFunction.Linear));

            Log.Line("Sample calculation:");
            nn.Calculate(input).Dump();

            int batch = 3;
            int trainNum = dataset.GetSampleCount(false);
            for (int epoch = 0; epoch < 10; epoch++)
            {
                for (int _ = 0; _ < trainNum / batch; _++)
                {
                    var inputs = new Matrix[batch];
                    var answers = new Matrix[batch];
                    double learningRate = 0.001;
                    for (int i = 0; i < batch; i++)
                    {
                        (inputs[i], answers[i]) = dataset.GetSample(NN.rand.Next(trainNum), false);
                    }
                    nn.Backpropagate(inputs, answers, learningRate);
                    Log.NativeLine($"{_}: ");
                    nn.EvaluateByLoss(inputs, answers);

                    if (false)
                        nn.EvaluateByQuestions(inputs, answers, (pred, ans) =>
                        {
                            return pred.MaxCell().r == ans.MaxCell().r;

                            /*
                            pred.Execute((d) => d < 0.5d ? 0d : 1d);
                            int counter = 0;
                            pred.RunFuncForEachCell((r, c, d) =>
                            {
                                if (d == ans[r, c])
                                    counter++;
                            });
                            return counter == ans.Rows;*/
                        });
                    if (_ % 1000 == 0)
                        nn.SaveToFile("nn.xml");
                }
            }
        }

        static void ImageLearnAdam()
        {
            var dataset = new ImageDataset("C:\\Users\\Kent2\\Desktop\\MyProgram\\WPF\\DeepLearningDraft\\archive");
            var order = new int[dataset.GetSampleCount(false)];
            for(int i = 0; i < order.Length; i++)
                order[i] = i;
            order = order.OrderBy(i => Guid.NewGuid()).ToArray();

            (var input, var desiredOutput) = dataset.GetSample(0, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample answer:");
            desiredOutput.Dump();

            var nn = NN.CreateFromFileOrNew("adam.xml", 8,
                LossFunction.CrossEntropy,
                new IntFuncPair(28 * 28, ActivationFunction.ReLu),
                new IntFuncPair(512, ActivationFunction.ReLu),
                new IntFuncPair(128, ActivationFunction.ReLu),
                new IntFuncPair(10, ActivationFunction.Linear));

            Log.Line("Sample calculation:");
            nn.Calculate(input).Dump();

            (Matrix input, Matrix output) GetRandomSample(bool test)
            {
                return dataset.GetSample(rand.Next(dataset.GetSampleCount(test)), test);
            }

            (var @in, var o) = GetRandomSample(false);
            Matrix[] prevMomentumFixedGrad = null;
            var prevRmsPropFixedGrad = prevMomentumFixedGrad;

            int epoch = 10000000;
            for (int i = 0; i < epoch; i++)
            {
                for(int n = 0; n < order.Length; n++)
                {
                    (var get, var output) = dataset.GetSample(order[n], false);
                    /*
                    (prevMomentumFixedGrad, prevRmsPropFixedGrad) =
                        nn.Adam(
                            prevMomentumFixedGrad, prevRmsPropFixedGrad, get, output,
                            0.9d, 0.01, 0.01, 1e-7);*/

                    /*
                    var diff = nn.LossDifferential(get, output);
                    prevMomentumFixedGrad = nn.Momentum(prevMomentumFixedGrad, diff, 0.9d);
                    nn.ApplyWeightAndBiasDiff(prevMomentumFixedGrad, 0.01d);
                    nn.EvaluateByLoss(new Matrix[] { get }, new Matrix[] { output });*/

                    var diff = nn.LossDifferential(get, output);
                    if(n > 20)
                    for (int j = 0; j < diff.Length; j++)
                    {
                        diff[j].Dump();
                    }
                    Matrix[] rate = null;
                    (rate, prevRmsPropFixedGrad) = nn.RmsProp(prevRmsPropFixedGrad, diff, 0.01d, 1e-8);
                    Log.Line(rate.ToString());
                    nn.ApplyWeightAndBiasDiff(diff, rate);
                    (get, output) = dataset.GetSample(rand.Next(dataset.GetSampleCount(true)), true);
                    nn.EvaluateByLoss(new Matrix[] { get }, new Matrix[] { output });

                    /*
                    var inputs = new Matrix[10];
                    var answers = new Matrix[10];
                    for (int k = 0; k < 10; k++)
                    {
                        (inputs[k], answers[k]) = dataset.GetSample(NN.rand.Next(dataset.GetSampleCount(false)), false);
                    }
                    nn.EvaluateByQuestions(inputs, answers, (pred, ans) =>
                    {
                        return pred.MaxCell().r == ans.MaxCell().r;
                    });*/
                }
                nn.SaveToFile("adam.xml");
            }
        }

        static void MathLearnAdam()
        {
            var dataset = new HalfAdderDataset();

            (var input, var desiredOutput) = dataset.GetSample(0, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample answer:");
            desiredOutput.Dump();

            var nn = NN.CreateFromFileOrNew("adam1.xml", 8,
                LossFunction.SumOfSquareError,
                new IntFuncPair(2, ActivationFunction.Sigmoid),
                new IntFuncPair(3, ActivationFunction.Sigmoid),
                new IntFuncPair(2, ActivationFunction.Sigmoid));

            Log.Line("Sample calculation:");
            nn.Calculate(input).Dump();

            (Matrix input, Matrix output) GetRandomSample(bool test)
            {
                return dataset.GetSample(rand.Next(dataset.GetSampleCount(test)), test);
            }

            (var @in, var o) = GetRandomSample(false);
            var prevMomentumFixedGrad = nn.LossDifferential(@in, o);
            var prevRmsPropFixedGrad = prevMomentumFixedGrad;

            int counter = 1000;
            for (int i = 0; i < counter; i++)
            {

                (var get, var output) = GetRandomSample(false);
                /*
                (prevMomentumFixedGrad, prevRmsPropFixedGrad) =
                    nn.Adam(
                        prevMomentumFixedGrad, prevRmsPropFixedGrad, get, output,
                        0.9d, 0.01, 0.01, 1e-7);*/

                /*
                var diff = nn.LossDifferential(get, output);
                prevMomentumFixedGrad = nn.Momentum(prevMomentumFixedGrad, diff, 0.9d);
                nn.ApplyWeightAndBiasDiff(diff, 0.01d);*/

                var diff = nn.LossDifferential(get, output);
                Matrix[] rate = null;
                (rate, prevRmsPropFixedGrad) = nn.RmsProp(prevRmsPropFixedGrad, diff, 0.01d, 1e-7);
                nn.ApplyWeightAndBiasDiff(diff, rate);
                Log.Line("Rate: " + rate);
                Log.NativeLine($"{i}: ");
                nn.EvaluateByLoss(new Matrix[] { get }, new Matrix[] { output });
            }

            nn.Dump();

            Log.Line("OUTPUT matrix = NN(INPUT matrix)");
            Log.Line("Compare OUTPUT and ANSWER");
            Log.Line("TEST RESULT: ");

            for (int idx = 0; idx < 4; idx++)
            {
                var data = dataset.GetSample(idx, false);
                Log.Line($"INPUT  No.{idx}");
                data.input.Dump();

                Log.Line($"OUTPUT No.{idx}");
                nn.Calculate(data.input).Dump();

                Log.Line($"ANSWER No.{idx}");
                data.desiredOutput.Dump();

                Log.NativeLine("\n");
            }
        }


        static void ImageTest()
        {
            var dataset = new ImageDataset("C:\\Users\\Kent2\\Desktop\\MyProgram\\WPF\\DeepLearningDraft\\archive");
            var nn = NN.CreateFromFileOrNew("nn.xml", 8,
                LossFunction.CrossEntropy,
                new IntFuncPair(28 * 28, ActivationFunction.ReLu),
                new IntFuncPair(512, ActivationFunction.ReLu),
                new IntFuncPair(128, ActivationFunction.ReLu),
                new IntFuncPair(10, ActivationFunction.Linear));

            int batch = dataset.GetSampleCount(true);
            var inputs = new Matrix[batch];
            var answers = new Matrix[batch];
            for (int i = 0; i < batch; i++)
            {
                var pair = dataset.GetSample(i, true);
                inputs[i] = pair.input;
                answers[i] = pair.desiredOutput;
            }
            nn.EvaluateByQuestions(inputs, answers, (output, answer) =>
            {
                return output.MaxCell().r == answer.MaxCell().r;
            });
        }

        static void HalfAdderTest()
        {
            var dataset = new HalfAdderDataset();
            (var input, var desiredOutput) = dataset.GetSample(0, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample output:");
            desiredOutput.Dump();

            var nn = new NN(4,
                LossFunction.SumOfSquareError,
                new IntFuncPair(2, ActivationFunction.Sigmoid),
                new IntFuncPair(3, ActivationFunction.Sigmoid),
                new IntFuncPair(2, ActivationFunction.Linear));

            // nn.Dump();
            for (int i = 0; i < 12000; i++)
            {
                int batch = 4;
                var inputs = new Matrix[batch];
                var answers = new Matrix[batch];
                double learningRate = 0.3;

                // Log.Line($"{i}th attempt, learning rate:{learningRate}");
                for (int j = 0; j < batch; j++)
                {
                    (inputs[j], answers[j]) = dataset.GetSample(i * batch + j, false);
                }
                nn.Backpropagate(inputs, answers, learningRate);
                nn.EvaluateByLoss(inputs, answers);
            }

            nn.SaveToFile("nnhalf.xml");

            nn.Dump();

            Log.Line("OUTPUT matrix = NN(INPUT matrix)");
            Log.Line("Compare OUTPUT and ANSWER");
            Log.Line("TEST RESULT: ");

            for (int idx = 0; idx < 4; idx++)
            {
                var data = dataset.GetSample(idx, false);
                Log.Line($"INPUT  No.{idx}");
                data.input.Dump();

                Log.Line($"OUTPUT No.{idx}");
                nn.Calculate(data.input).Dump();

                Log.Line($"ANSWER No.{idx}");
                data.desiredOutput.Dump();

                Log.NativeLine("\n");
            }
        }

        static void DifferentiableFuncTest()
        {
            int max = 10000;
            // f(d) = log(d)sin(d)^2 - 0.5
            //var dataset = new FuncDataset(max, (d) => Math.Log(d) * Math.Pow(Math.Sin(d), 2d) - 0.5d);
            var dataset = new FuncDataset(max, (d) => Math.Sin(d * Math.PI));
            (var input, var desiredOutput) = dataset.GetSample(100, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample answer:");
            desiredOutput.Dump();

            var nn = new NN(3,
                LossFunction.SumOfSquareError,
                new IntFuncPair(1, ActivationFunction.Sigmoid),
                new IntFuncPair(24, ActivationFunction.Sigmoid),
                new IntFuncPair(24, ActivationFunction.Sigmoid),
                new IntFuncPair(1, ActivationFunction.Linear));

            Log.Line("Sample output:");
            nn.Calculate(input).Dump();

            for (int j = 0; j < 20; j++)
            {
                int batch = 100;

                for (int header = 0; header < batch; header++)
                {
                    var amount = (max - header) / batch;
                    var inputs = new Matrix[amount];
                    var answers = new Matrix[amount];
                    double learningRate = 0.3;

                    for (int i = 0; i < amount; i++)
                    {
                        var index = header + i * batch;
                        (inputs[i], answers[i]) = dataset.GetSample(index, false);
                    }
                    nn.Backpropagate(inputs, answers, learningRate);
                    nn.EvaluateByLoss(inputs, answers);
                }
            }

            nn.Dump();

            double[] radians = {
        0,
        Math.PI / 10 / Math.PI,
        Math.PI / 10 * 2 / Math.PI,
        Math.PI / 10 * 3 / Math.PI,
        Math.PI / 10 * 4 / Math.PI,
        Math.PI / 10 * 5 / Math.PI,
        Math.PI / 10 * 6 / Math.PI,
        Math.PI / 10 * 7 / Math.PI,
        Math.PI / 10 * 8 / Math.PI,
        Math.PI / 10 * 9 / Math.PI,
        Math.PI / Math.PI };
            foreach (var radian in radians)
            {
                int intRad = (int)((double)max * radian);
                var data = dataset.GetSample(intRad, false);
                var output = nn.Calculate(data.input);

                Log.Line($"(Input, Output, Answer) = ({radian:0.000}, {output[0, 0]:0.000}, {data.desiredOutput[0, 0]:0.000})");
            }
        }

        static void AbsLearn()
        {
            int max = 10000;
            // f(d) = |x|
            var dataset = new FuncDataset(max, (d) => Math.Abs(d + 0.5f));
            (var input, var desiredOutput) = dataset.GetSample(100, false);

            Log.Line("Sample input:");
            input.Dump();

            Log.Line("Sample answer:");
            desiredOutput.Dump();

            var nn = new NN(3,
                LossFunction.SumOfSquareError,
                new IntFuncPair(1, ActivationFunction.Sigmoid),
                new IntFuncPair(12, ActivationFunction.Sigmoid),
                new IntFuncPair(12, ActivationFunction.Sigmoid),
                new IntFuncPair(1, ActivationFunction.Sigmoid));

            Log.Line("Sample output:");
            nn.Calculate(input).Dump();

            for (int j = 0; j < 20; j++)
            {
                int batch = 100;

                for (int header = 0; header < batch; header++)
                {
                    var amount = (max - header) / batch;
                    var inputs = new Matrix[amount];
                    var answers = new Matrix[amount];
                    double learningRate = 0.01;

                    for (int i = 0; i < amount; i++)
                    {
                        var index = header + i * batch;
                        (inputs[i], answers[i]) = dataset.GetSample(index, false);
                    }
                    nn.Backpropagate(inputs, answers, learningRate);
                    nn.EvaluateByLoss(inputs, answers);
                }
            }

            nn.Dump();

            for (var i = 0d; i < 1f; i += 0.1f)
            {
                int value = (int)(i * max);
                var data = dataset.GetSample(value, false);
                var output = nn.Calculate(data.input);

                Log.Line($"(Input, Output, Answer) = ({i:0.000}, {output[0, 0]:0.000}, {data.desiredOutput[0, 0]:0.000})");
            }
        }

        public static void WpfUiUpdate()
        {
            DispatcherFrame frame = new DispatcherFrame();
            var callback = new DispatcherOperationCallback(obj =>
            {
                ((DispatcherFrame)obj).Continue = false;
                return null;
            });
            Dispatcher.CurrentDispatcher.BeginInvoke(DispatcherPriority.Background, callback, frame);
            Dispatcher.PushFrame(frame);
        }
    }
}
