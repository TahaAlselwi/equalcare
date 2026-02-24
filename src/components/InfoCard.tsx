export default function InfoCard({
  title,
  children,
  spanFull = false,
}: {
  title: string;
  children: React.ReactNode;
  spanFull?: boolean;
}) {
  const showTitle = Boolean(title && title.trim());
  return (
    <section className={`card ${spanFull ? "span-full" : ""}`}>
      {showTitle ? <h3 className="card-title">{title}</h3> : null}
      <div className="card-body">{children}</div>
    </section>
  );
}
