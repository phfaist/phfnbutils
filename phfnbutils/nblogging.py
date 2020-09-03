import logging
import re
import string
import time

from collections import OrderedDict

from html import escape as escape_html

try:
    from IPython import get_ipython
    from IPython.display import display, HTML
except ImportError:
    get_ipython = lambda: None


class HTMLFormatter(logging.Formatter):
    """
    A formatter for log messages that generates HTML code.

    You can specify different HTML templates for different levels, and different
    CSS styles to be used by those templates.

    .. py::attribute: templates

       HTML templates to use for producing the log messages.  This attribute is
       a list of pairs `(levelno, html_template)` where `levelno` is a logging
       level such as `logging.DEBUG`.  The list must be sorted in increasing
       order of log level severity.

       The `html_template` is a string for use with `"...".format()` (with
       fields of the form `{field_name}`) where the field names may be any
       attributes of the `logging.LogRecord` object that is to be formatted.
       Additionally, a field name `css_style` can be used that is replaced by
       the CSS style that is relevant for this level (see `css_styles`
       attribute).

       The HTML template is fed into a custom :py:class:`string.Formatter` that
       ensures that all field substitutions are escaped suitably to be inserted
       in HTML code.  The formatter also offers some additional formats and
       conversions on top of the usual formatting specifications:

         - ``{fieldname:add-dot}``: the value is printed followed by a centered
           dot in a way that's suitable to be used at the beginning of a log
           message.  If the value is empty, the entire formatting instruction
           expands to an empty string and no dot is added, either.

         - ``{fieldname:add-brackets-dot}``: the value is put into square
           brackets, and then followed by a centered dot as with ``:add-dot``.
           If the value is empty, the entire formatting instruction expands to
           an empty string with no added dot.

         - ``{fieldname!E}``: special HTML characters in the formatted value are
           not escaped.  It is up to you to ensure that this produces valid and
           safe HTML code.

       The templates list does not have to specify a template for all logging
       levels.  When a HTML template is required for a given log level, we use
       the last template associated with a level that is not more severe than
       the requested level.  (Or we use the first template if the requested log
       level is of lesser severity than the first given template.)  For
       instance, you could use::

           html_formatter.templates = [
               (logging.INFO,
                '''<p style="margin:0pt;{css_style}">{message}</p>\n'''),
               (logging.ERROR,
                '''<p style="margin:0pt;{css_style}">*** {message} ***</p>\n'''),
           ]

       In which case the first template is used for all messages with levels
       DEBUG, INFO and WARNING, and the second is used for the ERROR and
       CRITICAL levels.


    .. py::attribute: css_par_styles
    
       This attribute should be a list of `(levelno, css_code)` with CSS styling
       code associated to level numbers.  The list works exactly as for the
       `templates` attributes.

       The css style code is then available to the template as the `css_style`
       field.

    .. py::attribute: 

    """
    def __init__(self, use_time=True, standard_names=None):
        super().__init__()

        self.use_time = use_time
        self.standard_names = standard_names
        if standard_names is None:
            self.standard_names = ['root', 'notebook']

        self.icons = OrderedDict([
            #(logging.ERROR, "🛑",),
            (logging.ERROR, "🚨",),
            (logging.WARNING, "⚠️",),
            (logging.INFO, "ℹ️",),
            (logging.DEBUG, "💨",),
        ])

        self.leading_parts = OrderedDict([
            ('icon', (self.format_html_leading_icon, " ") ),
            ('time', (self.format_html_leading_time, True) ),
            ('logger_name', (self.format_html_leading_loggername, True) ),
        ])
        self.leading_separator = "&nbsp;·&nbsp;"

        self.leading_final_separators = OrderedDict([
            #(logging.ERROR, "&nbsp;&nbsp;🚨&nbsp;&nbsp;"),
            (logging.WARNING, "&nbsp;·&nbsp;"),
        ])

        self.trailers = OrderedDict([
            (logging.ERROR, "&emsp;🚨"),
            (logging.WARNING, ""),
        ])

        self.css_par_styles = OrderedDict([
            (logging.ERROR,
             r'''font-weight:500;color:rgb(200,0,0)'''),
            (logging.WARNING,
             r'''font-weight:500;color:rgb(160,100,0)'''),
            (logging.INFO,
             r'''font-style:normal;color:rgb(40,40,80)'''),
            (logging.DEBUG,
             r'''font-style:normal;color:rgb(140,140,160)'''),
        ])

        self.css_message_styles = OrderedDict([
            (logging.ERROR,
             r'''font-size:1.1em'''),
            (logging.WARNING,
             r''''''),
        ])

        self.css_time = "font-size:0.8em;font-weight:300;"

        self.css_exception = "color:rgb(200,0,0);"
        self.css_stacktrace = "color:rgb(200,0,0);"

       

    def format_html_par(self, leading_html, message_html, trailer_html, record):
        """
        Wraps the given leading HTML code and the given message HTML code into an
        HTML `<p>` element that will represent the main part of the log record.
        Both `leading_html` and `message_html` arguments are assumed to be already properly HTML-escaped.
        """
        css_par_style = self._find_which_from_level(record.levelno, self.css_par_styles)
        css_message_style = self._find_which_from_level(record.levelno, self.css_message_styles)
        if leading_html:
            final_separator = self._find_which_from_level(record.levelno, self.leading_final_separators)
        else:
            final_separator = ''
        return \
            ('''<p style="margin:0pt;{css_par_style_html}">{leading_html}{maybe_leading_separator}'''
             '''<span style="{css_message_style_html}">{message_html}</span>{trailer_html}</p>\n''').format(
                 css_par_style_html=escape_html(css_par_style),
                 leading_html=leading_html,
                 maybe_leading_separator=final_separator,
                 css_message_style_html=escape_html(css_message_style),
                 message_html=message_html,
                 trailer_html=trailer_html
             )

    def format_html_leading_icon(self, record):
        """
        Get an emoji to associate with the given log level.  We search for an
        appropriate icon HTML code (such as emoji) in :py:attr:`icons`.

        This method is meant to be a possible leading part in
        :py:attr:`leading_parts`.
        """
        return self._find_which_from_level(record.levelno, self.icons)

    def format_html_leading_time(self, record):
        """
        Return the formatted time corresponding to the time the log record was
        emitted.  Returns an empty string if :py:attr:`use_time` is `False`.

        This method is meant to be a possible leading part in
        :py:attr:`leading_parts`.
        """
        if self.use_time:
            return '''<span style="{css_time_html}">{asctime_html}</span>'''.format(
                css_time_html=escape_html(self.css_time),
                asctime_html=escape_html(record.asctime),
            )
        return ""
    def format_html_leading_loggername(self, record):
        """
        Get the logger name, if it is not one of the boring "standard names" (see
        :py:attr:`standard_names`).  This method is meant to be a possible
        leading part in :py:attr:`leading_parts`.
        """
        if record.name not in self.standard_names:
            return '[' + escape_html(record.name) + ']'
        return ""

    def format_html_leading(self, record):
        """
        Put together all the leading elements to get the full leading HTML code.
        This concatenates all the elements given in :py:attr:`leading_parts`.
        """
        join_parts = []
        for part_fn, part_sep in self.leading_parts.values():
            x = part_fn(record)
            if x:
                join_parts.append(x)
                join_parts.append(self._get_the_leading_separator(part_sep))
        if not join_parts:
            return ""
        del join_parts[-1:] # no separator at the very end
        return "".join(join_parts)
    
    def _get_the_leading_separator(self, part_sep):
        if part_sep is None or part_sep is False:
            return ""
        if part_sep is True:
            return self.leading_separator
        return part_sep


    def format_html_exception(self, record):
        return '''<pre style="margin-top:0.5em;{css_exception_html}">{exc_text_html}</pre>\n'''.format(
            css_exception_html=escape_html(self.css_exception),
            exc_text_html=escape_html(record.exc_text),
        )
    def format_html_stacktrace(self, record):
        return '''<pre style="margin-top:0.5em;{css_stacktrace_html}">{formatted_stacktrace_html}</pre>\n'''.format(
            css_stacktrace_html=escape_html(self.css_stacktrace),
            formatted_stacktrace_html=escape_html(self.formatStack(record.stack_info)),
        )

    def format_html_trailer(self, record):
        """
        Returns what should be added at the end of a log message.  By default this is nothing.
        """
        return self._find_which_from_level(record.levelno, self.trailers)

    def format_html(self, record):
        """
        Return the full HTML code for this log record (including any possible
        exception information and stack trace).

        We obtain the "leading" and "trailer" parts with
        :py:meth:`format_html_leading()` and :py:meth:`format_html_trailer()`
        and feed that along with the record message to
        :py:meth:`format_html_par()`.  Additionally we append the results of
        :py:meth:`format_html_exception()` and/or
        :py:meth:`format_html_stacktrace()`, if appropriate.
        """
        s = self.format_html_par(
            self.format_html_leading(record),
            escape_html(record.message),
            self.format_html_trailer(record),
            record
        )
        if record.exc_text:
            s += self.format_html_exception(record)
        if record.stack_info:
            s += self.format_html_stacktrace(record)
        return s

    def _find_which_from_level(self, level, t_odic):
        # assuming t_odic is ordered in order of decreasing severity

        try:
            return next( fmt for k,fmt in t_odic.items() if k <= level )
        except StopIteration:
            return next(reversed(t_odic.values()))

    def formatTime(self, record):
        # see Python's Formatter.formatTime() source code
        ct = self.converter(record.created)
        return time.strftime(self.default_time_format, ct)

    def format(self, record):
        # adapted from Python's Formatter.format() source code

        record.message = record.getMessage()
        record.asctime = self.formatTime(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        return self.format_html(record)




def _ipython_display_html(s):
    display(HTML(s))


    
class HTMLHandler(logging.Handler):
    """
    A logging handler that displays messages in a Jupyter/IPython notebook with
    pretty formatting.

    You can specify a custom HTML formatter to produce the pretty messages with
    the `html_formatter` argument.  You can create and customize an instance of
    :py:class:`HTMLFormatter` that you can pass on to this argument.  You can
    also specify any `logging.Formatter` instance but then it's up to you to
    guarantee that it will generate valid HTML code by escaping the message
    correctly.

    This handler works by sending HTML code to the notebook for display via
    `IPython.display`.  You can override what is done with the HTML code by
    specifying a callable for `display_method`, in which case the callable is
    invoked with the HTML code as a single string argument.

    If `html_formatter=None`, then any additional keyword arguments are passed
    on to the constructor of :py:class:`HTMLFormatter`.  Otherwise you shouldn't
    specify additional keyword arguments.
    """
    def __init__(self, *, html_formatter=None, display_method=_ipython_display_html, **kwargs):
        super().__init__()
        self.display_method = display_method
        if html_formatter is not None:
            self.setFormatter(html_formatter)
            if kwargs:
                raise ValueError("Unexpected additional keyword arguments (you already "
                                 "provided a html_formatter: "+repr(kwargs))
        else:
            self.setFormatter(HTMLFormatter(**kwargs))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.display_method(msg)
        except Exception:
            self.handleError(record)


def setup(level=logging.DEBUG, *, no_debug_for=None, no_info_for=None, **kwargs):
    """
    Set up a basic logging environment and return a logger that you can use in
    the notebook.

    The root logger level is set to `level`.

    If `no_debug_for` is specified, it should be a tuple or a list of logger
    names whose individual levels should be set to `logging.INFO`.  You can use
    this to silence debugging messages in some external modules, such as 'qutip'
    or 'matplotlib'.  Similarly, with `no_info_for` you can specify a list of
    logger names whose levels should be set to `logging.WARNING`.

    Any `kwargs` are passed on to the :py:class:`HTMLHandler`.

    This method can also be used in non-ipython environments, in which case a
    basic stderr logger is set up via ``logging.basicConfig()``.
    """

    def fix_module_levels():
        # check that the user didn't specify by mistake
        # ``no_debug_for=('matplotlib'), ...`` because then it's the loggers 'm',
        # 'a', 't', .... that are silently and erroneously set to INFO level
        if isinstance(no_debug_for, str):
            raise ValueError(f"Expected tuple, list or iterable for no_debug_for, got string {no_debug_for!r}")
        if isinstance(no_info_for, str):
            raise ValueError(f"Expected tuple, list or iterable for no_debug_for, got string {no_info_for!r}")

        if no_debug_for:
            for loggername in no_debug_for:
                logging.getLogger(loggername).setLevel(logging.INFO)
        if no_info_for:
            for loggername in no_info_for:
                logging.getLogger(loggername).setLevel(logging.WARNING)
        
    if get_ipython() is None:
        # not running in IPython, use simple stderr output setup
        logging.basicConfig(level=level)
        logger = logging.getLogger("notebook")
        logger.debug("No IPython environment found, using standard stderr logging")
        fix_module_levels()
        return logger

    rootlogger = logging.getLogger()

    rootlogger.handlers = []
    # automatically creates a HTMLFormatter instance
    rootlogger.addHandler(HTMLHandler(**kwargs))

    rootlogger.setLevel(logging.DEBUG)

    fix_module_levels()
   
    return logging.getLogger("notebook")
