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
