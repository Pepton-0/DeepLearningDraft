using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    /// <summary>
    /// Deep learning neural network model
    /// </summary>
    public class NeuralNetwork
    {
        private readonly Node[][] Layers;

        /// <summary>
        /// From arguments create nodes of input layer, hidden layers, and output layer.<br/>
        /// Each node will be initialized with random weights, random bias and specified activation function.
        /// </summary>
        /// <param name="nodes"></param>
        public NeuralNetwork(params IntFuncPair[] pair)
        {
            if(pair.Length < 2)
            {
                throw new ArgumentException("Neural network must have at least input and output layers.");
            }

            Layers = new Node[pair.Length][];
            for (int i = 0; i < pair.Length; i++)
            {
                Layers[i] = new Node[pair[i].Integer];
                for (int j = 0; j < pair[i].Integer; j++)
                {
                    int numWeights = (i == 0) ? 0 : pair[i - 1].Integer;
                    Layers[i][j] = new Node(pair[i].Func, numWeights);
                }
            }
        }

        /// <summary>
        /// Initialized via argument layers.<br/>
        /// Used when loading from file or other sources.<br/>
        /// </summary>
        /// <param name="layers"></param>
        public NeuralNetwork(Node[][] layers)
        {
            this.Layers = layers;

            if(layers.Length < 2)
            {
                throw new ArgumentException("Neural network must have at least input and output layers.");
            }
        }

        /// <summary>
        /// Create copy of current neural network layers.
        /// </summary>
        /// <returns></returns>
        public Node[][] CloneNodes()
        {
            Node[][] clonedLayers = new Node[Layers.Length][];
            for (int i = 0; i < Layers.Length; i++)
            {
                clonedLayers[i] = new Node[Layers[i].Length];
                for (int j = 0; j < Layers[i].Length; j++)
                {
                    clonedLayers[i][j] = Layers[i][j].Clone();
                }
            }

            return clonedLayers;
        }

        /// <summary>
        /// Calculate output layer values from input values.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public double[] Calculate(params double[] inputs)
        {
            int inputLayerNum = Layers[0].Length;

            if (inputs.Length != inputLayerNum)
            {
                throw new ArgumentException("Input count must match input layer node count.");
            }

            for (int i = 0; i < inputLayerNum; i++)
            {
                Layers[0][i].Bias = inputs[i]; // Set input values to input layer nodes
            }

            return InternalCalculate(0, Layers[0].Select(v => v.GetActivity(new double[0])).ToArray());
        }

        /// <summary>
        /// Calculate the next layer's node array based on previous layer's node array.<br/>
        /// This continues until the last layer is reached.
        /// </summary>
        /// <param name="prevLayerIndex"></param>
        /// <param name="prevNodes"></param>
        /// <returns></returns>
        private double[] InternalCalculate(int prevLayerIndex, double[] prevNodes)
        {
            int layerIndex = prevLayerIndex + 1;

            if (layerIndex >= Layers.Length)
            {
                return prevNodes; // Return the last layer's output as output layer
            }

            double[] currentLayerNodeNum = new double[Layers[layerIndex].Length];
            for (int i = 0; i < Layers[layerIndex].Length; i++)
            {
                currentLayerNodeNum[i] = Layers[layerIndex][i].GetActivity(prevNodes);
            }

            return InternalCalculate(layerIndex, currentLayerNodeNum);
        }

        /// <summary>
        /// Calculate cost value from input values and expected output values.
        /// </summary>
        /// <returns></returns>
        public double CalculateCost(double[] inputs, double[] expectedOutputs)
        {
            var outputs = Calculate(inputs);
            double sum = 0d;
            for(int i = 0; i < outputs.Length; i++)
            {
                sum += Math.Pow(outputs[i] - expectedOutputs[i], 2);
            }

            return sum / outputs.Length;
        }

        public int GetInputLayerNodeNum()
        {
            return Layers[0].Length;
        }

        public int GetOutputLayerNodeNum()
        {
            return Layers[Layers.Length - 1].Length;
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
    }

    [DataContract]
    public class Node
    {
        /// <summary>
        /// From 0 to 1.
        /// </summary>
        [DataMember]
        public double[] Weights;

        /// <summary>
        /// When in input layer, this value is used as input value
        /// </summary>
        [DataMember]
        public double Bias;

        [DataMember]
        public ActivationFunction ActivationFunction;

        public Node(ActivationFunction func, int numWeights)
        {
            // TODO i wanna check if this is called with data serializer
            Log.Line("Called initialize function");
            SetRandomWeightsBiass(numWeights);
            this.ActivationFunction = func;
        }

        public Node(double[] weights, double bias, ActivationFunction activationFunction)
        {
            this.Weights = weights;
            this.Bias = bias;
            this.ActivationFunction = activationFunction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="prevLayerNodes">Can be null when the node is in input layer</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="NotSupportedException"></exception>
        public double GetActivity(double[] prevLayerNodes)
        {
            double sum = 0d;

            if(prevLayerNodes.Length != Weights.Length)
            {
                throw new ArgumentException("Previous layer nodes count must match current node weights count.");
            }

            if (prevLayerNodes != null)
            {
                for (int i = 0; i < prevLayerNodes.Length; i++)
                {
                    sum += prevLayerNodes[i] * Weights[i];
                }
            }

            sum += Bias;

            switch(ActivationFunction)
            {
                case ActivationFunction.ReLu:
                    return Math.Max(0d, sum);
                case ActivationFunction.ELU:
                    return sum >= 0d ? sum : 1d * (Math.Exp(sum) - 1d);
                default:
                    throw new NotSupportedException("Unsupported activation function.");
            }
        }

        public void SetRandomWeightsBiass(int numWeights)
        {
            Weights = new double[numWeights];
            for (int i = 0; i < numWeights; i++)
            {
                Weights[i] = App.rand.NextDouble();
            }
            Bias = App.rand.NextDouble();
        }

        public Node Clone()
        {
            return new Node((double[])this.Weights.Clone(), Bias, ActivationFunction);
        }
    }
}
