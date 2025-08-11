using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

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
            var network = new NeuralNetwork(
                new IntFuncPair(9, ActivationFunction.ELU),
                new IntFuncPair(6, ActivationFunction.ELU),
                new IntFuncPair(6, ActivationFunction.ELU),
                new IntFuncPair(3, ActivationFunction.ELU)
                );

            void Test(params double[] inputs)
            {
                double[] result = network.Calculate(inputs);
                string str = "";
                for (int i = 0; i < inputs.Length; i++)
                {
                    str+= $"{inputs[i]}";
                    if(i != inputs.Length - 1)
                    {
                        str += ", ";
                    }
                }
                string resultStr = "  -> ";
                for (int i = 0; i < result.Length; i++)
                {
                    resultStr += $"{result[i]}";
                    if (i != result.Length - 1)
                    {
                        resultStr += ", ";
                    }
                }
                Log.Line(str);
                Log.Line(resultStr);
                Log.Line("");
            }

            for (int i = 0; i < 5; i++)
            {
                double[] doubles = new double[9];
                for (int j = 0; j < doubles.Length; j++)
                {
                    doubles[j] = rand.NextDouble();
                }
                Test(doubles);
            }
        }
    }
}
