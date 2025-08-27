using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public class NeuralNetwork
    {
        /// <summary>
        /// n: numbers of nodes including input and output layers.<br/>
        /// Index from 0 (hidden layer after input layer) to n-1(output layer). <br/>
        /// Weight list for each layer.<br/>
        /// Array index is layer index.<br/>
        /// Matrix row axis is source node index.<br/>
        /// Matrix column axis is target node index. <br/>
        /// currentLayerOutputs : i+1 = activationFunc(layerWeights[i] * prevLayerOutputs + layerBiases[i])
        /// </summary>
        private readonly Matrix[] LayerWeights;

        /// <summary>
        /// Index from 0(hidden layer after input layer) to n-1(output layer).
        /// </summary>
        private readonly Matrix[] LayerBiases;

        /// <summary>
        /// Index from 0(hidden layer after input layer) to n-1(output layer).
        /// </summary>
        private readonly ActivationFunction[] ActivationFuncs;

        /// <summary>
        /// Number of layers include input and output layer.
        /// </summary>
        public readonly int LayerCount;

        public readonly int InputNodeCount;

        public readonly int OutputNodeCount;

        /// <summary>
        /// Numbers of pairs must be at least 2 (input layer and output layer).
        /// </summary>
        /// <param name="pairs"></param>
        /// <exception cref="ArgumentException"></exception>
        public NeuralNetwork(params IntFuncPair[] pairs) : this(
            pairs[0].Integer,
            pairs[pairs.Length - 1].Integer,
            new Matrix[pairs.Length - 1],
            new Matrix[pairs.Length - 1],
            new ActivationFunction[pairs.Length - 1])
        {
            if (pairs.Any(p => p.Integer <= 0))
            {
                throw new ArgumentException("All layers must have at least one node.");
            }

            for (int i = 1; i < pairs.Length; i++)
            {
                var prevPair = pairs[i - 1];
                var currentPair = pairs[i];
                var weightMatrix = new Matrix(currentPair.Integer, prevPair.Integer);
                weightMatrix.Randomize();
                var biasMatrix = Matrix.FromRVector(new double[currentPair.Integer]);
                biasMatrix.Randomize();
                LayerWeights[i - 1] = weightMatrix;
                LayerBiases[i - 1] = biasMatrix;
                ActivationFuncs[i - 1] = currentPair.Func;
            }
        }

        /// <summary>
        /// Base constructor
        /// </summary>
        /// <param name="layerWeights"></param>
        /// <param name="layerBiases"></param>
        /// <param name="activationFuncs"></param>
        public NeuralNetwork(int inputNum, int outputNum, Matrix[] layerWeights, Matrix[] layerBiases, ActivationFunction[] activationFuncs)
        {
            this.LayerWeights = layerWeights;
            this.LayerBiases = layerBiases;
            this.ActivationFuncs = activationFuncs;

            this.LayerCount = layerWeights.GetLength(0) + 1;
            this.InputNodeCount = inputNum;
            this.OutputNodeCount = outputNum;

            Log.Line($"{inputNum} : {outputNum}");
        }

        /// <summary>
        /// Calculate row matrix outputs from inputs
        /// </summary>
        /// <param name="inputs">Should be (input numbers, 1)</param>
        /// <returns>Matrix (output numbers, 1)</returns>
        public Matrix Calculate(Matrix inputs)
        {
            Matrix prevActivations = inputs;

            for (int i = 0; i < LayerWeights.GetLength(0); i++)
            {
                prevActivations = InternalCalculate(i, prevActivations);
            }

            return prevActivations;
        }

        /// <summary>
        /// Calculate row matrix outputs from inputs, and return all layer node activations.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public Matrix[] CalculateAll(Matrix inputs)
        {
            var matrices = new Matrix[LayerCount - 1];

            Matrix prevActivations = inputs;

            for (int i = 0; i < LayerWeights.GetLength(0); i++)
            {
                prevActivations = InternalCalculate(i, prevActivations);
                matrices[i] = prevActivations;
            }

            return matrices;
        }

        /// <summary>
        /// Calculate cost from outputs and actual answers
        /// </summary>
        /// <param name="output"></param>
        /// <param name="answers"></param>
        /// <returns>Cost >= 0</returns>
        public double CalculateCostFromOutputs(Matrix outputs, Matrix answers)
        {
            var matrix = outputs - answers;
            double sum = 0;
            matrix.FilterFunc((from) =>
            {
                sum += from * from;
                return from;
            });

            return sum;
        }

        public double CalculateCostFromInputs(Matrix inputs, Matrix answers)
        {
            return CalculateCostFromOutputs(Calculate(inputs), answers);
        }

        public double CalculateAvgCostFromInputs(Matrix[] inputs, Matrix[] answers)
        {
            if(inputs.Length != answers.Length)
            {
                throw new ArgumentException("Inputs and answers length must be same.");
            }

            var costs = new double[inputs.Length];
            for(int i = 0; i < inputs.Length; i++)
            {
                costs[i] = CalculateCostFromInputs(inputs[i], answers[i]);
            }
            Array.Sort(costs);
            return costs[costs.Length/2];
        }

        /// <summary>
        /// Calculate the activation matrix of the index's layer from last layer's activations
        /// </summary>
        /// <param name="index"></param>
        /// <param name="prevActivations"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        private Matrix InternalCalculate(int index, Matrix prevActivations)
        {
            var func = ActivationFuncs[index];
            // row matrix; activation list of current index layer nodes
            var matrix = LayerWeights[index] * prevActivations + LayerBiases[index];
            matrix.FilterFunc((from) =>
            {
                switch (func)
                {
                    case ActivationFunction.ReLu:
                        return Math.Max(0d, from);
                    case ActivationFunction.ELU:
                        return from >= 0d ? from : 1d * (Math.Exp(from) - 1d);
                    case ActivationFunction.Sigmoid:
                        return NMath.Sigmoid(from);
                    default:
                        throw new NotImplementedException($"This function is not implemented: {func}");
                }
            });
            return matrix;
        }

        /// <summary>
        /// First matrices are layer weights, second matrices are layer biases.
        /// </summary>
        /// <returns></returns>
        public (Matrix[], Matrix[]) CloneMatrices()
        {
            return (LayerWeights.Select(m => m.Clone()).ToArray(),
                    LayerBiases.Select(m => m.Clone()).ToArray());
        }

        /// <summary>
        /// Calculate gradient for the specific inputs matrix(any, 1).<br/>
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns>Last cost</returns>
        public double CalculateGradient(Matrix[] inputs, Matrix[] answers)
        {
            double oridinalCost = CalculateAvgCostFromInputs(inputs, answers);

            // List up each partial differential for each weight and bias, which is cost(inputs + delta(w)) - cost(inputs) / delta(w)
            var allValues = new SharedRMatrix(LayerWeights.Union(LayerBiases).ToArray());
            var diffs = new Matrix(allValues.Rows, 1); // add this to weights and biases to minimize cost

            bool Approximately(double a, double b)
            {
                return Math.Abs(a - b) <= 0.0001;
            }

            double CalculatePartialDifferential(int index, SharedRMatrix w)
            {
                var tmpValue = w[index, 0];
                var prevDerivative = 0d;
                var nextDerivative = double.PositiveInfinity;
                var h = 2d;
                int counter = 0;

                do
                {
                    counter++;
                    if(counter >= 500)
                    {
                        Log.Line("Might be infinite loop in CalculatePartialDifferential");
                    }
                    h /= 2d;
                    prevDerivative = nextDerivative;
                    w[index, 0] = tmpValue + h;
                    nextDerivative = (CalculateAvgCostFromInputs(inputs, answers) - oridinalCost) / h;
                    // Log.Line($"Calculated {index + 1}/{allValues.rows} partial differential: {nextDerivative}");
                } while (!Approximately(prevDerivative, nextDerivative));

                w[index, 0] = tmpValue;
                return nextDerivative;
            }

            for (int i = 0; i < allValues.Rows; i++)
            {
                var diff = CalculatePartialDifferential(i, allValues);
                diffs[i, 0] = -diff; // TODO i wanna add some noise for escaping local minimum
                // Log.Line($"Calculated {i + 1}/{allValues.rows} partial differential: {diff}");
            }

            double lastCost = 0d;
            double nextCost = double.PositiveInfinity;
            double distance = 1.0d;

            do
            {
                if (lastCost > nextCost)
                    distance /= 2d;
                else
                    distance *= 1.5d; // when candidate goes too far, come back a bit

                lastCost = nextCost;
                for(int i = 0; i < diffs.Rows; i++)
                {
                    allValues[i, 0] = diffs[i, 0] * distance;
                }
                nextCost = CalculateAvgCostFromInputs(inputs, answers);
            } while(!Approximately(lastCost, nextCost));

            return nextCost;
        }

        public double Backpropagate(Matrix[] inputs, Matrix[] answers)
        {
            var weightClone = new Matrix[LayerWeights.Length];
            var weightDiffClones = new Matrix[LayerWeights.Length];
            // var biasDiffClones = new Matrix[LayerBiases.Length];
            var results = new Matrix[inputs.Length][];
            for(int i = 0; i < inputs.Length; i++)
            {
                results[i] = CalculateAll(inputs[i]);
            }

            for(int i = 0; i < LayerWeights.Length; i++)
            {
                weightClone[i] = LayerWeights[i].Clone();
                weightClone[i].FilterFunc((from) => from >= 0 ? 1 : -1);
                var biasClone = LayerBiases[i].Clone();
                weightDiffClones[i] = new Matrix(weightClone[i].Rows, weightClone[i].Columns);
                // biasDiffClones[i] = new Matrix(biasClone.rows, biasClone.columns);
            }

            for(int i = 0; i < results.Length; i++)
            {
                var result = results[i];
                var answer = answers[i];
                var desiredActivations = answer - result[result.Length - 1];

                for (int j = LayerWeights.Length - 1; j > 0; j--)
                {
                    var tmp = new Matrix(desiredActivations.Rows, desiredActivations.Columns);
                    var weights = LayerWeights[j];
                    // var biases = LayerBiases[j]; // 1 ~ 0 - any = 
                    // element is plus -> need to be increased
                    // element is minus -> need to be decreased


                    // act > 0 && weight > 0 -> increase weight
                    // act > 0 && weight < 0 -> decrease weight
                    // act < 0 && weight > 0 -> decrease weight
                    // act < 0 && weight < 0 -> increase weight

                    for(int row = 0; row < weights.Rows; row++)
                    {
                        for(int col = 0; col < weights.Columns; col++)
                        {
                            var d = weightClone[j][row, col] * desiredActivations[col, 0];
                            weightDiffClones[j][row, col] += d;
                            tmp[col, 0] += d;
                        }
                    }
                    desiredActivations = tmp;
                }
            }

            for (int i = 0; i < LayerWeights.Length; i++)
            {
                var w = weightDiffClones[i];
                // var b = biasDiffClones[i];
                w.FilterFunc((from) => from / results.Length * 0.1d);
                LayerWeights[i] += w;
                //b.FilterFunc((from) => from / results.Length);
            }

            return CalculateAvgCostFromInputs(inputs, answers);
        }
    }

    public struct IntFuncPair
    {
        public int Integer;
        public ActivationFunction Func;

        public IntFuncPair(int integer, ActivationFunction func)
        {
            this.Integer = integer;
            this.Func = func;
        }
    }

    public enum ActivationFunction
    {
        ReLu,
        ELU,
        Sigmoid,
    }
}
