using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Shapes;

namespace DeepLearningDraft.NN
{
    public class NN
    {
        /// <summary>
        /// Layer node matrix: row matrix
        /// Expanded layer node matrix: row matrix(nodes+1, 1) for bias
        /// 
        /// Layer weight matrix: matrix(target node amount, prev node amount)
        /// Expanded weight & bias matrix: matrix(target node amount, prev node amount + 1) for bias
        /// bias is in matrix(any, 0)
        /// </summary>
        private readonly Matrix[] WeightsAndBiases;

        /// <summary>
        /// Layer numbers include hidden & output layers, exclude input layer
        /// </summary>
        private readonly int LayerCount;

        private delegate double ActivationFunc(double d);
        private readonly ActivationFunc[] ActivationFuncs;
        private readonly ActivationFunc[] ActivationFuncDiffs;

        public NN(double biasRange, params IntFuncPair[] pairs) : this(Pair2WeightsAndBiases(pairs, biasRange), pairs.Select(s => s.Func).ToArray())
        {
            if (pairs.Any(p => p.Integer <= 0))
            {
                throw new ArgumentException("All layers must have at least one node.");
            }
        }

        public NN(Matrix[] weightsAndBiases, ActivationFunction[] funcs)
        {
            this.WeightsAndBiases = weightsAndBiases;
            this.LayerCount = weightsAndBiases.Length;

            ActivationFuncs = funcs.Select<ActivationFunction, ActivationFunc>((f) =>
            {
                switch (f)
                {
                    case ActivationFunction.Sigmoid:
                        return Mathf.Sigmoid;
                    case ActivationFunction.ReLu:
                        return Mathf.ReLU;
                    default:
                        throw new NotImplementedException("The function is not implemented yet");
                }
            }).ToArray();

            ActivationFuncDiffs = funcs.Select<ActivationFunction, ActivationFunc>((f) =>
            {
                switch (f)
                {
                    case ActivationFunction.Sigmoid:
                        return Mathf.SigmoidDiff;
                    case ActivationFunction.ReLu:
                        return Mathf.ReLUDiff;
                    default:
                        throw new NotImplementedException("The function is not implemented yet");
                }
            }).ToArray();
        }

        private static Matrix[] Pair2WeightsAndBiases(IntFuncPair[] pairs, double biasRange)
        {
            var matrices = new Matrix[pairs.Length - 1]; // Exclude input layer
            for (int i = 1; i < pairs.Length; i++)
            {
                var pair = pairs[i];
                var weight_bias_layer = new Matrix(pair.Integer, pairs[i - 1].Integer + 1);
                weight_bias_layer.Randomize();
                weight_bias_layer.FillFunc((r, c) => (c == 0 ? biasRange : 1) * weight_bias_layer[r,c]);
                matrices[i - 1] = weight_bias_layer;
            }
            return matrices;
        }

        public void Dump()
        {
            for (int i = 0; i < LayerCount; i++)
            {
                Log.Line($"--{i} layer dump--");
                WeightsAndBiases[i].Dump();
                Trace.WriteLine("");
            }
        }

        private static readonly Matrix DummyMatrix1 = Matrix.Fill1(1, 1);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="prevNodesAndBiases">Should be row matrix which has all previous node activations</param>
        /// <param name="index"></param>
        /// <returns></returns>
        private Matrix CalculateNextLayer(Matrix prevNodes, int index)
        {
            var currentNodes = WeightsAndBiases[index] * Matrix.CombineRow(DummyMatrix1, prevNodes);
            currentNodes.Execute((d) => ActivationFuncs[index](d));

            return currentNodes;
        }

        private (Matrix activated, Matrix nonActivated) CalculateNextLayer_WithNonActivated(Matrix prevNodes, int index)
        {
            var currentNodes = WeightsAndBiases[index] * Matrix.CombineRow(DummyMatrix1, prevNodes);
            var activated = currentNodes.Clone();
            currentNodes.Execute((d) => ActivationFuncs[index](d));

            return (activated, currentNodes);
        }

        public Matrix Calculate(Matrix inputs)
        {
            var matrix = inputs;
            for(int i = 0; i < LayerCount; i++)
            {
                matrix = CalculateNextLayer(matrix, i);
            }
            return matrix;
        }

        /// <summary>
        /// From the same size matrix: outputs & answers, calculate loss functions
        /// Loss = 1/2 * Σ(i=0, i<outputs.Length){(output[i,0] - answer[i,0])^2}
        /// dL/dOutput[any,0] = output - answer
        /// </summary>
        /// <param name="outputs"></param>
        /// <param name="answers"></param>
        /// <returns></returns>
        public double LossFromOutputs(Matrix outputs, Matrix answers)
        {
            var matrix = outputs - answers;
            matrix.Execute((d) => d * d);
            var f = matrix * Matrix.Fill1(1, matrix.Rows);
            
            return f[0, 0];
        }

        public double LossFromInputs(Matrix inputs, Matrix answers)
        {
            return LossFromOutputs(Calculate(inputs), answers);
        }

        public double LossAvgFromInputs(Matrix[] inputs, Matrix[] answers)
        {
            if(inputs.Length != answers.Length)
            {
                throw new ArgumentException("inputs and answers dont have the same length");
            }

            double[] costs = new double[inputs.Length];
            for(int i = 0; i <  inputs.Length; i++)
            {
                costs[i] = LossFromInputs(inputs[i], answers[i]);
            }
            Array.Sort(costs);
            {
                double sum = 0;
                int counter = 0;

                for (int i = costs.Length / 4; i < costs.Length - costs.Length / 4; i++)
                {
                    counter++;
                    sum += costs[i];
                }

                if (counter <= 3 || double.IsInfinity(sum))
                    goto CalculationFailed;

                return sum / counter;
            }

        CalculationFailed:
            return costs[costs.Length / 2];
        }

        /// <summary>
        /// Return all the node values exclude input nodes
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix[] CalculateAllNodes(Matrix input)
        {
            var matrices = new Matrix[LayerCount];
            for (int i = 0; i < LayerCount; i++)
            {
                matrices[i] = CalculateNextLayer((i - 1 <= -1 ? input : matrices[i - 1]), i);
            }

            return matrices;
        }

        public (Matrix[] nodes , Matrix[] nonActivatedNodes) CalculateAllNodesWithNonActivated(Matrix input)
        {
            var nodes = new Matrix[LayerCount + 1];
            var nonActivatedNodes = new Matrix[LayerCount + 1];
            nodes[0] = input;
            nonActivatedNodes[0] = null;
            for (int i = 1; i <= LayerCount; i++)
            {
                (nodes[i], nonActivatedNodes[i]) = 
                    CalculateNextLayer_WithNonActivated(nodes[i - 1], i-1);
            }

            return (nodes, nonActivatedNodes);
        }

        /// <summary>
        /// Calculate all the gradient differential for each weights and biases to fit with given input & answer
        /// </summary>
        /// <param name="input"></param>
        /// <param name="answer"></param>
        /// <returns></returns>
        public Matrix[] LossDifferential(Matrix input, Matrix answer)
        {
            var (nodes, nonActivatedNodes) = CalculateAllNodesWithNonActivated(input);
            /*
            Log.Line("Nodes dump");
            for (int i = 0; i < nodes.Length; i++)
            {
                nodes[i].Dump();
            }*/
            var diffCollection = new Matrix[WeightsAndBiases.GetLength(0)];

            // dl_dz = layerNode - answer; row matrix
            // sigmoid(nodes) = nodes.Execute(x => ActivationFuncDiffs[i](x))

            Matrix lastBiasDiffs = null;
            Matrix lastWeightDiffs = null;

            // Calculate all the diff from output layer to first hidden layer
            for (int layerIndex = LayerCount; layerIndex >= 1; layerIndex--)
            {
                var indexFromHidden = layerIndex - 1;
                var weightAndBias = WeightsAndBiases[indexFromHidden];
                var diffMatrix = diffCollection[indexFromHidden] = new Matrix(weightAndBias.Rows, weightAndBias.Columns);
                var layerNode = nodes[layerIndex];

                // with current loss function, the diff of loss func should be following formula
                // dl_dz
                Matrix dL_dNodes;

                if (layerIndex == LayerCount)
                { // if output layer 

                    // dL_dBiases = dL_dNodes ○ ActivationFuncDiff(nonActivatedNodes)
                    var I = nonActivatedNodes[layerIndex].Clone();
                    I.Execute(d => ActivationFuncDiffs[indexFromHidden](d));

                    dL_dNodes = layerNode - answer;
                    lastBiasDiffs = dL_dNodes;
                    lastBiasDiffs.HadamarProduct(I);

                    lastWeightDiffs = lastBiasDiffs * nodes[layerIndex - 1].Transpose();
                }
                else
                { // if hidden layer

                    // dL_dNodes(current) = weightMatrix(next).Transpose() * dL_dBias(next)
                    // = (dL_dBias(next).Transpose() * weightMatrix(next)).Transpose()
                    var lastWeightAndBias = WeightsAndBiases[indexFromHidden + 1];
                    dL_dNodes =
                        (lastBiasDiffs.Transpose() * Matrix.SelectColumn(lastWeightAndBias, 1, lastWeightAndBias.Columns))
                        .Transpose();

                    var I = nonActivatedNodes[layerIndex].Clone();
                    I.Execute(d => ActivationFuncDiffs[indexFromHidden](d));
                    lastBiasDiffs = dL_dNodes;
                    lastBiasDiffs.HadamarProduct(I);

                    lastWeightDiffs = lastBiasDiffs * nodes[layerIndex - 1].Transpose();
                }

                lastBiasDiffs.RunFuncForEachCell((r, c, d) => { diffMatrix[r, c] = d; });
                lastWeightDiffs.RunFuncForEachCell((r, c, d) => { diffMatrix[r, c + 1] = d; });

                /*
                Log.Line("Dump trace in GradientDifferential");
                
                lastBiasDiffs.Dump();
                lastWeightDiffs.Dump();
                diffMatrix.Dump();
                diffCollection[indexFromHidden].Dump();*/
            }

            return diffCollection;
        }

        /// <summary>
        /// d > 0
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="answers"></param>
        /// <param name="learningRate"></param>
        public void GradientDecent(Matrix[] inputs, Matrix[] answers, double learningRate)
        {
            Matrix[] diffSum = null;

            for (int i = 0; i < inputs.Length; i++)
            {
                var diff = LossDifferential(inputs[i], answers[i]);
                if (diffSum == null)
                    diffSum = diff;
                else
                {
                    for(int j = 0; j < diffSum.Length; j++)
                    {
                        diffSum[j] += diff[j];
                    }
                }
            }
            //diffSum = LossDifferential(inputs[0], answers[0]);

            for (int i = 0; i < diffSum.Length; i++)
            {
                var layer = diffSum[i];
                layer.Execute((d) => d / inputs.Length * learningRate);
                WeightsAndBiases[i] -= layer;
                // Log.Line($"Add diff to layer[{i}]");
                // layer.Dump();
            }
        }

        /// <summary>
        /// Dump accuracy info based on loss of the given inputs & answers
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="answers"></param>
        public void EvaluateByLoss(Matrix[] inputs, Matrix[] answers)
        {

            var loss = LossAvgFromInputs(inputs, answers);
            /*
            for(int i = 0; i < WeightsAndBiases.Length; i++)
            {
                WeightsAndBiases[i].Dump();
            }*/
            Log.Line($"Loss score: {loss}");
        }

        public void EvaluateByQuestions(Matrix[] inputs, Matrix[] answers, Func<Matrix, Matrix, bool> checker)
        {
            int sum = 0;
            int qualified = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var answer = answers[i];

                var output = Calculate(input);
                var check = checker(output, answer);
                if (check)
                    qualified++;
                sum++;
            }

            Log.Line($"{{Correct/Questions}} = {qualified.ToString("000")}/{sum.ToString("000")} = rate: {((double)qualified/sum).ToString("0.000")}");
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
