using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningDraft
{
    /// <summary>
    /// Logger service that automatically writes member name which uses log function.
    /// </summary>
    public class Log
    {
        private static string Category(string caller)
        {
            return $"[{caller}]";
        }

        private static string GetTrace()
        {
#if DEBUG
            var methodInfo = new StackTrace().GetFrame(2).GetMethod();
            return methodInfo.ReflectedType.Name; // Class name
#else
            return "[LOG]";
#endif
        }

        public static void Line(string arg)
        {
            Trace.WriteLine(arg, Category(GetTrace()));
        }

        public static void Line(object obj)
        {
            Trace.WriteLine(obj, Category(GetTrace()));
        }

        public static void LongTrace(string longTrace)
        {
            Trace.WriteLine("---Long trace print---", Category(GetTrace()));
            Trace.Indent();
            Trace.WriteLine(longTrace);
            Trace.Unindent();
        }
    }
}
