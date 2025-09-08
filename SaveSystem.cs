using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace DeepLearningDraft
{
    public class SaveSystem
    {
        private static readonly string SaveDir = Path.GetDirectoryName(AppContext.BaseDirectory);

        /// <summary>
        /// Save class object to XML file.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj"></param>
        /// <param name="filename"></param>
        public static void Save<T>(T obj, string filename) where T : class
        {
            try
            {
                string path = Path.Combine(SaveDir, filename);
                DataContractSerializer serializer = new DataContractSerializer(typeof(T));
                XmlWriterSettings settings = new XmlWriterSettings { Encoding = Encoding.UTF8 };
                using (var writer = XmlWriter.Create(path, settings))
                {
                    Log.Line($"Save {nameof(T)} to {path}.");
                    serializer.WriteObject(writer, obj);
                }
            }
            catch (Exception ex)
            {
                Log.Line($"Error saving to {filename} from {nameof(T)}.");
                Log.LongTrace(ex.ToString());
            }
        }

        /// <summary>
        /// Save the buffer to specific file in the following format:<br/>
        /// byte[] of int: size of buffer<br/>
        /// byte[] of byte[]: buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        public static void SaveBuffer(byte[] buffer, string filename)
        {
            string path = Path.Combine(SaveDir, filename);
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write))
            {
                using (BufferedStream bs = new BufferedStream(fs))
                {
                    int size = buffer.Length;
                    byte[] sizeData = BitConverter.GetBytes(size);
                    bs.Write(sizeData, 0, sizeData.Length);
                    bs.Write(buffer, 0, buffer.Length);
                    bs.Flush();
                }
            }
        }

        public static byte[] LoadBuffer(string filename)
        {
            string path = Path.Combine(SaveDir, filename);
            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
            {
                using (BufferedStream bs = new BufferedStream(fs))
                {
                    var sizeData = new byte[sizeof(int)];
                    bs.Read(sizeData, 0, sizeData.Length);
                    
                    var buffer = new byte[BitConverter.ToInt32(sizeData, 0)];
                    bs.Read(buffer, 0, buffer.Length);

                    return buffer;
                }
            }
        }

        public static T Load<T>(string filename) where T : class
        {
            try
            {
                string path = Path.Combine(SaveDir, filename);
                DataContractSerializer serializer = new DataContractSerializer(typeof(T));
                using (var reader = XmlReader.Create(filename))
                {
                    Log.Line($"Load {nameof(T)} from {path}.");
                    return (T)serializer.ReadObject(reader);
                }
            }
            catch (Exception ex)
            {
                Log.Line($"Error loading from {filename} to {nameof(T)}. Return null instead.");
                Log.LongTrace(ex.ToString());
                return null;
            }
        }
    }
}
