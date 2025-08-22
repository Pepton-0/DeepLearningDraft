using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    public class NN
    {
        /// <summary>
        /// n: numbers of nodes including input and output layers.<br/>
        /// Index from 0 (hidden layer after input layer) to n-1(output layer). <br/>
        /// Weight list for each layer.<br/>
        /// Array index is layer index.<br/>
        /// Matrix row axis is source node index
        /// Matrix column axis is target node index.
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

        public NN(params IntFuncPair[] pairs) : this(
            pairs[0].Integer,
            pairs[pairs.Length - 1].Integer,
            new Matrix[pairs.Length - 1],
            new Matrix[pairs.Length - 1],
            new ActivationFunction[pairs.Length - 1])
        {
            if(pairs.Length < 2)
            {
                throw new ArgumentException("Neural network must have at least input, output layers.");
            }

            if(pairs.Any(p => p.Integer <= 0))
            {
                throw new ArgumentException("All layers must have at least one node.");
            }

            for (int i = 1; i < pairs.Length - 1; i++)
            {
                var prevPair = pairs[i- 1];
                var currentPair = pairs[i];
                var weightMatrix = new Matrix(currentPair.Integer, prevPair.Integer);
                weightMatrix.Randomize();
                var biasMatrix = Matrix.ToRVector(new double[currentPair.Integer]);
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
        public NN(int inputNum, int outputNum, Matrix[] layerWeights, Matrix[] layerBiases, ActivationFunction[] activationFuncs)
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
            
            for(int i = 0; i < LayerWeights.GetLength(0) - 1; i++)
            {
                prevActivations = InternalCalculate(i, prevActivations);
            }

            return prevActivations;
        }

        /// <summary>
        /// Calculate cost from outputs and actual answers
        /// </summary>
        /// <param name="output"></param>
        /// <param name="answers"></param>
        /// <returns>Cost >= 0</returns>
        public double CalculateCost(Matrix outputs, Matrix answers)
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
            return CalculateCost(Calculate(inputs), answers);
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
    }
}
