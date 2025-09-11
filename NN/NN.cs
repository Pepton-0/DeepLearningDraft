using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DeepLearningDraft
{
    /// <summary>
    /// Feedfoward neural network and deep learning mechanism
    /// </summary>
    public class NN
    {
        private static bool LOG = false;

        /// <summary>
        /// Use this for every random calculation
        /// </summary>
        public static readonly Random rand = new Random();

        /// <summary>
        /// Used for multithread calculation
        /// </summary>
        public static readonly int NUM_WORKER_THREAD = Environment.ProcessorCount;

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

        private delegate void ActivationFunc(Matrix d);

        /// <summary>
        /// Used for every layer from first hidden to output layer
        /// </summary>
        private readonly ActivationFunc[] ActivationFuncs;

        /// <summary>
        /// Differential for activation functions
        /// </summary>
        private readonly ActivationFunc[] ActivationDiffFuncs;

        private delegate double LossFunc(Matrix output, Matrix answer);

        /// <summary>
        /// Loss function
        /// </summary>
        private readonly LossFunc LossFunc_;
        private delegate Matrix LossDiffFunc(Matrix output, Matrix answer);

        /// <summary>
        /// Differential of loss function
        /// </summary>
        private readonly LossDiffFunc LossDiffFunc_;

        public NN(double biasRange, LossFunction loss, params IntFuncPair[] pairs) : this(Pair2WeightsAndBiases(pairs, biasRange), pairs.Select(s => s.Func).ToArray(), loss)
        {
            if (pairs.Any(p => p.Integer <= 0))
            {
                throw new ArgumentException("All layers must have at least one node.");
            }
        }

        public NN(Matrix[] weightsAndBiases, ActivationFunction[] funcs, LossFunction loss)
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
                    case ActivationFunction.Linear:
                        return Mathf.Linear;
                    default:
                        throw new NotImplementedException("The function is not implemented yet");
                }
            }).ToArray();

            ActivationDiffFuncs = funcs.Select<ActivationFunction, ActivationFunc>((f) =>
            {
                switch (f)
                {
                    case ActivationFunction.Sigmoid:
                        return Mathf.SigmoidDiff;
                    case ActivationFunction.ReLu:
                        return Mathf.ReLUDiff;
                    case ActivationFunction.Linear:
                        return Mathf.LinearDiff;
                    default:
                        throw new NotImplementedException("The function is not implemented yet");
                }
            }).ToArray();

            switch (loss)
            {
                case LossFunction.SumOfSquareError:
                    this.LossFunc_ = Mathf.Loss_SumOfSquareError;
                    this.LossDiffFunc_ = Mathf.LossDiff_SumOfSquareError;
                    break;
                case LossFunction.CrossEntropy:
                    this.LossFunc_ = Mathf.Loss_CrossEntropy;
                    this.LossDiffFunc_ = Mathf.LossDiff_CrossEntropy;
                    break;
                default:
                    throw new NotImplementedException("The function is not implemented yet");
            }
        }

        private static Matrix[] Pair2WeightsAndBiases(IntFuncPair[] pairs, double biasRange)
        {
            var matrices = new Matrix[pairs.Length - 1]; // Exclude input layer
            for (int i = 1; i < pairs.Length; i++)
            {
                var pair = pairs[i];
                var weight_bias_layer = new Matrix(pair.Integer, pairs[i - 1].Integer + 1);

                var wSamples = Normal.Samples(
                    rand,
                    0,
                    Math.Sqrt(2d / (pairs[i - 1].Integer + pair.Integer))
                    ).GetEnumerator();
                for (int j = 0; j < 10; j++)
                {
                    wSamples.MoveNext();
                    Log.Line($"wSample:${j},{pairs[i - 1].Integer}:{wSamples.Current}");
                }
                weight_bias_layer.FillFunc((r, c) =>
                {
                    if (c == 0)
                    { // is bias
                        return 0;
                    }
                    else // is weight
                    {
                        wSamples.MoveNext();
                        return wSamples.Current;
                    }
                });

                matrices[i - 1] = weight_bias_layer;
            }
            return matrices;
        }

        /// <summary>
        /// Show all the weights and biases of each layer
        /// </summary>
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
        /// Calculate the index layer's node activations from previous nodes
        /// </summary>
        /// <param name="prevNodes">Should be row matrix which has all previous node activations</param>
        /// <param name="index"></param>
        /// <returns></returns>
        private Matrix CalculateNextLayer(Matrix prevNodes, int index)
        {
            var currentNodes = WeightsAndBiases[index] * Matrix.CombineRow(DummyMatrix1, prevNodes);
            ActivationFuncs[index](currentNodes);

            return currentNodes;
        }

        private (Matrix activated, Matrix nonActivated) CalculateNextLayer_WithNonActivated(Matrix prevNodes, int index)
        {
            var nonActivated = WeightsAndBiases[index] * Matrix.CombineRow(DummyMatrix1, prevNodes);
            var activated = nonActivated.Clone();
            // activated.Execute((d) => ActivationFuncs[index](d));
            ActivationFuncs[index](activated);

            return (activated, nonActivated);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix Calculate(Matrix input)
        {
            var matrix = input;
            for (int i = 0; i < LayerCount; i++)
            {
                matrix = CalculateNextLayer(matrix, i);
            }
            return matrix;
        }

        public double LossFromOutputs(Matrix output, Matrix answer)
        {
            return LossFunc_(output, answer);
        }

        public double LossFromInput(Matrix input, Matrix answer)
        {
            return LossFromOutputs(Calculate(input), answer);
        }

        public double LossAvgFromInputs(Matrix[] inputs, Matrix[] answers)
        {
            if (inputs.Length != answers.Length)
            {
                throw new ArgumentException("inputs and answers dont have the same length");
            }

            var commonBatchNum = inputs.Length / NUM_WORKER_THREAD; // 0~any
            var lastRemainderIdx = inputs.Length % NUM_WORKER_THREAD - 1;
            Vector2[] begin_end_arr = new Vector2[NUM_WORKER_THREAD];
            int lastIndex = 0;
            for (int i = 0; i < begin_end_arr.Length; i++)
            {
                var reminder = lastRemainderIdx >= i;
                var batchNum = commonBatchNum + (reminder ? 1 : 0);
                begin_end_arr[i] = new Vector2(lastIndex, lastIndex + batchNum);
                lastIndex += batchNum;
            }
            var partialSum = Task.WhenAll(Enumerable.Range(0, NUM_WORKER_THREAD).Select(n => Task.Run(() =>
            {
                var localSum = 0d;
                int begin = (int)begin_end_arr[n].X;
                int end = (int)begin_end_arr[n].Y;
                for (int i = begin; i < end; i++)
                {
                    localSum += LossFromInput(inputs[i], answers[i]);
                }

                return localSum;
            }))).GetAwaiter().GetResult();

            double sum = 0d;
            for (int i = 0; i < partialSum.Length; i++)
            {
                sum += partialSum[i];
            }

            return sum / inputs.Length;
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

        public (Matrix[] nodes, Matrix[] nonActivatedNodes) CalculateAllNodesWithNonActivated(Matrix input)
        {
            var activatedNodes = new Matrix[LayerCount + 1];
            var nonActivatedNodes = new Matrix[LayerCount + 1];
            activatedNodes[0] = input;
            nonActivatedNodes[0] = null;
            for (int i = 1; i <= LayerCount; i++)
            {
                (activatedNodes[i], nonActivatedNodes[i]) =
                    CalculateNextLayer_WithNonActivated(activatedNodes[i - 1], i - 1);

                if (LOG)
                {
                    Log.Line($"{i}th activatedNodes, nonActivatedNodes");
                    activatedNodes[i].Dump();
                    nonActivatedNodes[i].Dump();
                }
            }

            return (activatedNodes, nonActivatedNodes);
        }

        /// <summary>
        /// Calculate all the gradient differential for each weights and biases to fit with given input & answer
        /// </summary>
        /// <param name="input"></param>
        /// <param name="answer"></param>
        /// <returns></returns>
        public Matrix[] LossDifferential(Matrix input, Matrix answer)
        {
            var (activatedNodes, nonActivatedNodes) = CalculateAllNodesWithNonActivated(input);
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
                var layerNode = activatedNodes[layerIndex];

                // with current loss function, the diff of loss func should be following formula
                // dl_dz
                Matrix dL_dNodes;

                if (LOG)
                {
                    Log.Line($"Calculate diff for {indexFromHidden}");
                }

                if (layerIndex == LayerCount)
                { // if output layer 
                  // dL_dNodes = layerNode - answer;
                    dL_dNodes = LossDiffFunc_(layerNode, answer);
                    if (LOG)
                    {
                        Log.Line("dL_dNodes = LossDiffFunc(layerNode, answer)");
                        dL_dNodes.Dump();
                        layerNode.Dump();
                        answer.Dump();
                    }
                }
                else
                { // if hidden layer

                    // dL_dNodes(current) = weightMatrix(next).Transpose() * dL_dBias(next)
                    // = (dL_dBias(next).Transpose() * weightMatrix(next)).Transpose()
                    var lastWeightAndBias = WeightsAndBiases[indexFromHidden + 1];
                    dL_dNodes =
                        (lastBiasDiffs.Transpose() * lastWeightAndBias.SelectColumn(1, lastWeightAndBias.Columns))
                        .Transpose();
                    if (LOG)
                    {
                        Log.Line($"dL_dNodes[{indexFromHidden}] = dL_dBias[{indexFromHidden} + {1}] * dL_dWeights[{indexFromHidden} + 1]");
                        dL_dNodes.Dump();
                        lastBiasDiffs.Transpose().Dump();
                        lastWeightAndBias.SelectColumn(1, lastWeightAndBias.Columns).Dump();
                    }
                }

                // dL_dBiases = dL_dNodes ○ ActivationFuncDiff(nonActivatedNodes)
                var I = nonActivatedNodes[layerIndex].Clone();
                //I.Execute(d => ActivationDiffFuncs[indexFromHidden](d));
                ActivationDiffFuncs[indexFromHidden](I);
                //var I = layerNode.Clone();
                //I.HadamarProduct(Matrix.Fill1(I.Rows, I.Columns) - I);
                lastBiasDiffs = dL_dNodes.Clone();
                lastBiasDiffs.HadamarProduct(I);

                if (LOG)
                {
                    Log.Line("dL_dBiases = dL_dNodes 〇 ActivationFuncDiff(nonActivatedNodes)");
                    lastBiasDiffs.Dump();
                    dL_dNodes.Dump();
                    I.Dump();
                    nonActivatedNodes[layerIndex].Dump();
                }

                lastWeightDiffs = lastBiasDiffs * activatedNodes[layerIndex - 1].Transpose();
                if (LOG)
                {
                    Log.Line("dL_dWeights = dL_dBiases * activatedNodes[n - 1].Transpose()");
                    lastWeightDiffs.Dump();
                    lastBiasDiffs.Dump();
                    activatedNodes[layerIndex - 1].Transpose().Dump();

                }

                lastBiasDiffs.CopyTo(0, 0, diffMatrix);
                lastWeightDiffs.CopyTo(0, 1, diffMatrix);
                //lastBiasDiffs.RunFuncForEachCell((r, c, d) => { diffMatrix[r, c] = d; });
                //lastWeightDiffs.RunFuncForEachCell((r, c, d) => { diffMatrix[r, c + 1] = d; });
            }

            return diffCollection;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="answers"></param>
        /// <param name="learningRate"></param>
        public void Backpropagate(Matrix[] inputs, Matrix[] answers, double learningRate)
        {
            Matrix[] diffSum = null;

            /** Task table
             * CPU0: OOOOOO
             * CPU1: OOOOOO
             * CPU2: OOOOO
             */

            var commonBatchNum = inputs.Length / NUM_WORKER_THREAD; // 0~any
            var lastRemainderIdx = inputs.Length % NUM_WORKER_THREAD - 1;
            Vector2[] begin_end_arr = new Vector2[NUM_WORKER_THREAD];
            int lastIndex = 0;
            for (int i = 0; i < begin_end_arr.Length; i++)
            {
                var reminder = lastRemainderIdx >= i;
                var batchNum = commonBatchNum + (reminder ? 1 : 0);
                begin_end_arr[i] = new Vector2(lastIndex, lastIndex + batchNum);
                lastIndex += batchNum;
                // Log.Line($"Batch{begin_end_arr}:" + begin_end_arr[i]);
            }

            var partialDiff = Task.WhenAll(Enumerable.Range(0, NUM_WORKER_THREAD).Select(n => Task.Run(() =>
            {
                Matrix[] localDiff = null;
                var vec = begin_end_arr[n];
                int begin = vec.X;
                int end = vec.Y;
                for (int i = begin; i < end; i++)
                {
                    var diff = LossDifferential(inputs[i], answers[i]);
                    if (localDiff == null)
                        localDiff = diff;
                    else
                    {
                        for (int j = 0; j < localDiff.Length; j++)
                            localDiff[j] += diff[j];
                    }
                }

                return localDiff;
            }))).GetAwaiter().GetResult();

            for (int i = 0; i < partialDiff.GetLength(0); i++)
            {
                var diff = partialDiff[i];

                if (diffSum == null)
                    diffSum = diff;
                else if (diff != null)
                {
                    for (int j = 0; j < diff.GetLength(0); j++)
                    {
                        diffSum[j] += diff[j];
                    }
                }
            }

            /*
            for (int i = 0; i < inputs.Length; i++)
            {
                if (LOG)
                {
                    Log.Line($"Calculate loss differential for {i}th sample");
                }
                var diff = LossDifferential(inputs[i], answers[i]);
                if (diffSum == null)
                    diffSum = diff;
                else
                {
                    for (int j = 0; j < diffSum.Length; j++)
                    {
                        diffSum[j] += diff[j];
                    }
                }
            }*/

            for (int k = 0; k < diffSum.Length; k++)
            {
                var layer = diffSum[k];
                layer.Execute((d) => d / inputs.Length * learningRate);
                WeightsAndBiases[k] -= layer;
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
            var commonBatchNum = inputs.Length / NUM_WORKER_THREAD; // 0~any
            var lastRemainderIdx = inputs.Length % NUM_WORKER_THREAD - 1;
            Vector2[] begin_end_arr = new Vector2[NUM_WORKER_THREAD];
            int lastIndex = 0;
            for (int i = 0; i < begin_end_arr.Length; i++)
            {
                var reminder = lastRemainderIdx >= i;
                var batchNum = commonBatchNum + (reminder ? 1 : 0);
                begin_end_arr[i] = new Vector2(lastIndex, lastIndex + batchNum);
                lastIndex += batchNum;
            }
            var partialQualified = Task.WhenAll(Enumerable.Range(0, NUM_WORKER_THREAD).Select(n => Task.Run(() =>
            {
                var localQualified = 0;
                var vec = begin_end_arr[n];
                int begin = vec.X;
                int end = vec.Y;
                for (int i = begin; i < end; i++)
                {
                    bool satisfied = checker(Calculate(inputs[i]), answers[i]);
                    if (satisfied)
                        localQualified++;
                }
                return localQualified;
            }))).GetAwaiter().GetResult();

            int qualified = 0;
            for (int i = 0; i < partialQualified.Length; i++)
            {
                qualified += partialQualified[i];
            }
            int sum = inputs.Length;

            Log.Line($"{{Correct/Questions}} = {qualified.ToString("000")}/{sum.ToString("000")} = rate: {((double)qualified / sum).ToString("0.000")}");
        }

        public void SaveToFile(string filename)
        {
            /*
            var arr = new double[WeightsAndBiases.Sum(m => m.Rows * m.Columns)];
            int i = 0;
            for (int l = 0; l < WeightsAndBiases.Length; l++)
            {
                var matrix = WeightsAndBiases[l];
                var row = matrix.Rows;
                var col = matrix.Columns;
                for (int r = 0; r < row; r++)
                {
                    for (int c = 0; c < col; c++)
                    {
                        arr[i] = matrix[r, c];
                        i++;
                    }
                }
            }*/
            // TODO why this causes error?
            var arr = WeightsAndBiases.Select(r => r.ToByte1DimArr()).ToArray();
            var buffer = new byte[arr.Length];
            Buffer.BlockCopy(arr, 0, buffer, 0, buffer.Length);

            SaveSystem.SaveBuffer(buffer, filename);
        }

        public static NN CreateFromFileOrNew(string filename, double biasRange, LossFunction loss, params IntFuncPair[] pairs)
        {
            var arr = SaveSystem.Load<double[]>(filename);
            if (arr == null)
            {
                Log.Line("Created new one");
                return new NN(biasRange, loss, pairs);
            }
            else
            {
                var matrices = new Matrix[pairs.Length - 1];
                int i = 0;
                for (int l = 1; l < pairs.Length; l++)
                {
                    var row = pairs[l].Integer;
                    var col = pairs[l - 1].Integer + 1;
                    var matrix = matrices[l - 1] = new Matrix(row, col);
                    for (int r = 0; r < row; r++)
                    {
                        for (int c = 0; c < col; c++)
                        {
                            matrix[r, c] = arr[i];
                            i++;
                        }
                    }
                }
                /*
                var matrices = new Matrix[pairs.Length - 1];
                var idx = 0;
                for(int l = 1; l < pairs.Length ; l++)
                {
                    var row = pairs[l].Integer;
                    var col = pairs[l - 1].Integer + 1;
                    var partialBytes = new byte[row * col * sizeof(double)];
                    Array.Copy(arr, idx, partialBytes, 0, partialBytes.Length);
                    matrices[l - 1] = Matrix.FromByte1DimArr(row, col, partialBytes);

                    idx += partialBytes.Length;
                }*/
                var activationFuncs = pairs.Select(pair => pair.Func).ToArray();
                return new NN(matrices, activationFuncs, loss);
            }
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
        Sigmoid,
        Linear,
    }

    public enum LossFunction
    {
        SumOfSquareError,
        CrossEntropy
    }
}